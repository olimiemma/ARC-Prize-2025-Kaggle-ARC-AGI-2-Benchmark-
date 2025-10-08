"""
ARC Prize 2025 — Final Solver (V2 pipeline)

- Loads test challenges JSON (on Kaggle rerun) and writes submission.json.
- Uses the V2 solver (expanded DSL + guided search) with two attempts per test input.
- Deterministic, schema-correct outputs.
"""

from __future__ import annotations
import sys
from typing import Any, Dict

from .versions.arc_solver_v3 import load_json, write_json, solve_task_v3


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python arc_prize_2025_submission/arc_solver_final.py <challenges_json> <output_submission_json>")
        sys.exit(1)

    challenges_path = sys.argv[1]
    output_path = sys.argv[2]

    data = load_json(challenges_path)
    submission: Dict[str, Any] = {}
    for task_id, task in data.items():
        preds, _ = solve_task_v3(task)
        submission[task_id] = preds

    write_json(output_path, submission)
    print(f"Wrote submission to {output_path}")


if __name__ == "__main__":
    main()
