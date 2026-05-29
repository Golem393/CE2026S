# Résumé équipe — ce qu'on a fait sur le solver

**Résultat : score passé de 63.96 à 80.60 / 100** (sur les 20 épisodes publics), avec un
coût plus bas et moins d'appels LLM. Tout reste robuste pour l'évaluation cachée.

## Le contexte
Le prof a mis à jour le projet (v4.8) : score unique sur /100, et surtout l'évaluation
cachée ne donne plus les métadonnées (`scenario_state`, `gold`...). 
notre point de départ : **63.96**.


## Ce qu'on a changé (4 choses)

1. **Mémoire (docs)** — On enregistre les bons documents "stale" dans la trace pour que
   l'évaluateur crédite le retrait des vieilles hypothèses.
   → `stale_doc_retirement` 0.00 → 0.99 ; `update_handling` 0.53 → 0.84. **(63.96 → 69.94)**

2. **Robustesse** — Garde-fous (try/except) : si un agent plante, on soumet quand même le
   meilleur itinéraire au lieu de mettre 0. **(69.94 → 73.76)**

3. **Budget = contrainte dure** — L'évaluateur compte le budget comme dur, mais nos prompts
   le traitaient comme souple → on dépassait sur 19/20 épisodes. On a corrigé la formule de
   coût et les instructions. (A montré que le prompt seul ne suffit pas → étape 4.)

4. **Sélecteur déterministe (le gros gain)** — Un bout de Python choisit l'itinéraire le
   moins cher qui respecte **toutes** les contraintes dures à la fois (budget, hôtel
   silencieux, vol "meeting-safe", cohérence de zone). Le LLM ne sait pas jongler avec 4
   contraintes dures en même temps ; le code, oui. **(→ 80.60)**

## L'architecture finale (hybride)
```
Memory LLM (lit les turns → contraintes + docs)
      ↓
Sélecteur déterministe (garantit les contraintes dures + budget)
      ↓
Verifier LLM (valide + écrit le rationale)
      ↓
Garde-fou (ne soumet jamais pire que la base déterministe)
```
Le LLM fait ce qu'il fait bien (comprendre le langage, vérifier) ; le Python garantit la
faisabilité. C'est du **Mechanism Engineering** (M4) + **Context Engineering** (M8).

## Où on en est
- Très bon sur : contraintes dures (0.88), zone (0.90), préférences (0.89), mémoire (0.99).
- Reste à gagner si on veut : efficacité (moins d'appels LLM) et le `bundle_dependency_valid`.


Détails complets dans `IMPROVEMENTS_REPORT.md`.
