
# Baseline LLM Prompt Notes

Goal: build a compact single-agent planner.

Required behaviors:
- respect spoken rules
- use profile, venue, and city notes when relevant
- return strict JSON
- keep memory report concise
- do not hallucinate unavailable IDs

Likely weakness to preserve:
- limited retirement behavior
- weaker verifier-style checking
- tendency to over-carry context
