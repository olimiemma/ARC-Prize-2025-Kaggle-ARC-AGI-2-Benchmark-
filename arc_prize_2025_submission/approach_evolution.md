Approach Evolution (Plan)

V1: Baseline
- Implement IO, evaluation harness, deterministic fallbacks.
- Add minimal DSL ops (rotate, flip, transpose, palette_map, pad/crop, CC labeling).
- Feature extraction and very small beam search (B≈8, D≈3) with exact verification.

V2: Guided Search
- Expand DSL; add parameter constraints from features (translate, tile, component copy/place).
- Canonicalize operation sequences; add stronger pruning heuristics.
- Improve two-attempt orchestration with explicit prior selection.

V3: Optimization and Optional Learned Guidance
- Profile and optimize hot paths; cache intermediate computations.
- Add small packaged ranker (optional) for tie-breaking; keep deterministic fallbacks.
- Tune beam/depth per task based on quick features within time budgets.

Finalization
- Harden submission path; verify schema; ensure deterministic runs and logging.
- Document method, constraints, and limitations for Kaggle submission.

