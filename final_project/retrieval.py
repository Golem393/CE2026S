from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_runner import LLMRunner


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def cosine_similarity(a: List[float], b: List[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def lexical_score(query: str, text: str) -> float:
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q:
        return 0.0
    return len(q & t) / len(q)


class RetrievalCorpus:
    def __init__(self, docs: List[Dict[str, Any]], *, cache_dir: str | Path):
        self.docs = [dict(doc) for doc in docs]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _doc_signature(self) -> str:
        blob = json.dumps(
            [{"doc_id": doc["doc_id"], "text": doc["text"]} for doc in self.docs],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

    def _cache_path(self, model: str) -> Path:
        safe_model = model.replace("/", "_")
        return self.cache_dir / f"memory_corpus_{safe_model}_{self._doc_signature()}.json"

    def _load_cached_embeddings(self, model: str) -> Optional[Dict[str, List[float]]]:
        path = self._cache_path(model)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def _save_cached_embeddings(self, model: str, vectors: Dict[str, List[float]]) -> None:
        self._cache_path(model).write_text(json.dumps(vectors))

    def ensure_doc_embeddings(self, runner: LLMRunner, model: str) -> Dict[str, Any]:
        cached = self._load_cached_embeddings(model)
        if cached and all(doc["doc_id"] in cached for doc in self.docs):
            for doc in self.docs:
                doc["embedding"] = cached[doc["doc_id"]]
            return {"usage": runner.empty_usage(), "cache_hit": True}

        payload = runner.embed_texts(model=model, texts=[doc["text"] for doc in self.docs])
        vectors = {}
        for doc, vector in zip(self.docs, payload["vectors"]):
            doc["embedding"] = vector
            vectors[doc["doc_id"]] = vector
        self._save_cached_embeddings(model, vectors)
        return {"usage": payload["usage"], "cache_hit": False}

    def _filtered_docs(
        self,
        *,
        memory_type: str | None = None,
        city: str | None = None,
        traveler_id: str | None = None,
        family: str | None = None,
        include_stale: bool = True,
    ) -> List[Dict[str, Any]]:
        out = []
        for doc in self.docs:
            if memory_type and doc.get("memory_type") != memory_type:
                continue
            if city and doc.get("city") not in {None, city}:
                continue
            if traveler_id and doc.get("traveler_id") not in {None, traveler_id}:
                continue
            if family and doc.get("family") not in {None, family}:
                continue
            if not include_stale and doc.get("status") == "stale":
                continue
            out.append(dict(doc))
        return out

    def search(
        self,
        *,
        runner: LLMRunner,
        query: str,
        strategy: str,
        top_k: int,
        embedding_model: str | None = None,
        memory_type: str | None = None,
        city: str | None = None,
        traveler_id: str | None = None,
        family: str | None = None,
        include_stale: bool = True,
    ) -> Dict[str, Any]:
        usage = runner.empty_usage()
        docs = self._filtered_docs(
            memory_type=memory_type,
            city=city,
            traveler_id=traveler_id,
            family=family,
            include_stale=include_stale,
        )
        for doc in docs:
            doc["lexical_score"] = lexical_score(query, f"{doc['doc_id']} {doc['text']} {' '.join(doc.get('tags', []))}")
        docs.sort(key=lambda row: row["lexical_score"], reverse=True)

        if strategy == "embedding" and embedding_model and docs:
            doc_embed = self.ensure_doc_embeddings(runner, embedding_model)
            usage = runner.combine_usages(usage, doc_embed["usage"])
            query_embed = runner.embed_texts(model=embedding_model, texts=[query])
            usage = runner.combine_usages(usage, query_embed["usage"])
            query_vec = query_embed["vectors"][0]
            for doc in docs:
                doc["embedding_score"] = cosine_similarity(query_vec, next(d["embedding"] for d in self.docs if d["doc_id"] == doc["doc_id"]))
                doc["hybrid_score"] = 0.25 * doc["lexical_score"] + 0.75 * doc["embedding_score"]
            docs.sort(key=lambda row: row["hybrid_score"], reverse=True)
        else:
            for doc in docs:
                doc["embedding_score"] = 0.0
                doc["hybrid_score"] = doc["lexical_score"]

        ranked = docs[:top_k]
        for rank, doc in enumerate(ranked, start=1):
            doc["rank"] = rank
            doc.pop("embedding", None)
        return {
            "query": query,
            "strategy": strategy,
            "results": ranked,
            "usage": usage,
        }
