ARC Prize 2025 — Solver Architecture (Step 2)

Architecture Selection and Rationale

- Core approach: Hybrid symbolic solver using a concise micro-DSL of grid operations, feature-driven priors, and guided beam search with exact verification on train pairs. Produce two diverse programs per task to populate attempt_1 and attempt_2.
- Rationale: Literature and competition reports consistently show that exact verification with search over compositional grid ops aligns with ARC’s discrete, exact-match nature. Neuro-symbolic and LLM-augmented methods provide strong priors but add packaging/training constraints. This design prioritizes robustness, determinism, and Kaggle feasibility while allowing later integration of learned guidance.
- Evidence: See literature/metrics_summary.txt and literature/comparative_matrix.txt for strong outcomes using enumeration + selection; ARC 2024 technical report highlights hybrid/search pipelines; NSA shows learned priors reducing search while keeping exact verification.

System Components

1) Data and IO
- Loader for challenges JSON (training/evaluation/test) with schema validation and stable ordering of test inputs per task.
- Submission writer for submission.json adhering strictly to Kaggle schema.

2) Feature Extraction
- Per-pair features: grid sizes, palette histograms and deltas, CC counts/sizes, symmetry scores (rot/flip), border/frame signatures, centroid shifts.
- Aggregation across train pairs: invariants (histogram preservation modulo relabel), consistent size deltas/ratios, stable CC mappings, presence of new colors.

3) Micro-DSL (Pure, Deterministic)
- Geometric: rotate(k), flip(axis), transpose, translate(dx,dy,fill), pad_crop(dh,dw,fill), tile(hr,wr).
- Component: label_cc(), extract/largest_component(), map_per_component(fn), copy/place with mask.
- Palette: palette_map({src→dst}), recolor(condition).
- Selection/Structure: detect frames/borders, checker/stripe masks, symmetry axes.
- Notes: Operations are parameterized; parameters come from feature-driven candidate sets to constrain the search.

4) Program Induction (Search Engine)
- Guided beam search with beam size B (e.g., 8–16) and depth D (e.g., 3–5). Multiple roots corresponding to distinct priors.
- Node expansion constrained by features: e.g., rotation in {0,90,180,270}; flips {H,V}; translation by estimated (dx,dy) from components; palette maps derived from pairwise color correspondences.
- Early pruning: reject partial programs that already violate any train pair; maintain operation-effect canonicalization to avoid redundant sequences (e.g., rotate×4 ≡ identity).
- Tie-breaking: prefer shorter programs and simpler ops; secondary heuristic ranks if multiple fits remain.

5) Exact Verifier
- Applies a candidate program to all train inputs and checks exact equality to outputs. Only exact fits are accepted.
- Optional learned tie-breaker (deferred): a small deterministic scorer may be added later, packaged offline.

6) Two-Attempt Orchestrator
- Attempt_1 (palette-first track): Start from palette invariants and structure-preserving operations; explore recolor/rotate/flip/transpose/pad_crop; translate if CC-inferred.
- Attempt_2 (geometry-first track): Start from size-change and symmetry hypotheses; explore tile/translate/component copy-and-place; allow palette_map only if new colors appear or histogram changes.
- Fallbacks: If no program fits, emit shape-safe, deterministic fallbacks inferred from train (e.g., pad/crop policy, identity, or dominant-color fill as last resort) to maintain valid submission entries.

7) Evaluation Harness (Offline)
- Mirrors Kaggle scoring for the evaluation split: for each task and test input, success if attempt_1 OR attempt_2 equals ground truth. Supports timing and regression checks.

8) Logging and Determinism
- Per-task logs: prior chosen, features summary, candidates explored, final program, timing, and fallback reason (if any).
- Determinism: fixed seeds; stable sort orders; no nondeterministic ops; identical outputs across runs.

Search Strategy Details

- Priors selection:
  - Histogram preserved (up to relabel) → palette-first.
  - Consistent size deltas/ratios → geometry-first (pad/crop/tile).
  - Strong symmetry scores → symmetry-first (rotate/flip/transpose roots).
  - New colors appear → recolor/masking track allowed.
- Parameter generation:
  - Rotations {0,90,180,270}; flips {H,V}; transpose {T, none} considered within combination caps.
  - Translate(dx,dy) from CC centroid deltas; restrict to small ranges derived from train pairs.
  - Palette maps inferred from color co-occurrence across pairs; allow one-to-one first, then many-to-one as needed.
- Beam and depth:
  - Default B=12, D=4 as a starting point; adapt per task using quick heuristics (e.g., if size matches, deprioritize tile).
  - Hard per-task time budget to respect Kaggle’s 12h wall-clock.

Resource Planning (Kaggle Constraints)

- No internet; code must be self-contained. If any learned component is used, it must be bundled via Kaggle Datasets/Models and loaded deterministically.
- Memory: DSL operations work on small grids; careful vectorization keeps CPU usage low; GPU not required.
- Runtime: Feature extraction O(HW) per example; search bounded by beam×depth×param caps with aggressive pruning.
- Determinism: Ensure reproducibility across Kaggle reruns.

Risks and Mitigations

- Search explosion → Mitigate with feature-constrained params, early pruning, operation canonicalization, tight beams/depths.
- Overfitting to train → Prefer shorter, structure-preserving programs when features allow; require exact fit across all pairs.
- Pathological tasks → Maintain diverse priors and robust fallbacks to always submit valid attempts.
- Integration errors → Use evaluation harness locally on the evaluation split to catch schema/timing issues before submitting.

Next Steps

1) Scaffold repository structure and stubs for: data loader, feature extractor, DSL ops, search engine, verifier, two-attempt orchestrator, evaluation harness, and submission writer.
2) Implement V1 baseline: feature extraction, minimal DSL subset, small beam search, exact verification, deterministic fallbacks, and submission writer.
3) Iterate with profiling/tuning; expand DSL ops and priors; improve pruning and tie-breaking. Optionally add a small learned ranker packaged offline.

