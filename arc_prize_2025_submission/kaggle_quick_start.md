Kaggle Quick Start

What this provides

- Production entrypoint: arc_prize_2025_submission/arc_solver_final.py (uses V3 pipeline).
- Research docs: research_synthesis.md, architecture_design.md.
- Versioned implementations: versions/arc_solver_v1.py … v3.py.

Usage (local)

- For local experiments on the evaluation split, use CLI in versions/arc_solver_v1.py, versions/arc_solver_v2.py, or versions/arc_solver_v3.py to:
  - Load evaluation challenges/solutions.
  - Run the solver and compute exact-match with the harness logic.
  - Dump submission.json for smoke tests.

Usage (Kaggle Code Competition)

- In the Kaggle notebook, ensure you read the test challenges JSON (Kaggle provides the hidden file on rerun).
- Invoke arc_solver_final.py with: python arc_solver_final.py <test_challenges_json> submission.json
- Ensure deterministic behavior and that submission.json has two attempts per test input for all tasks.

Notes

- No internet on Kaggle; any models must be packaged via Kaggle Datasets/Models.
- Keep total runtime under the 12-hour limit; prefer CPU-friendly code.
