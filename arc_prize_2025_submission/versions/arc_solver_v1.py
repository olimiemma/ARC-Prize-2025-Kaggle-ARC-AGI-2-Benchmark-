"""
ARC Solver V1

Baseline with minimal micro-DSL, small guided search, exact verification, and
two-attempt strategy. Provides a CLI for local evaluation and submission file
generation.
"""

from __future__ import annotations
import json
import sys
from typing import Any, Dict, List, Tuple, Optional


# -----------------
# IO Utilities
# -----------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


# -----------------
# Grid Ops (Minimal DSL)
# -----------------

Grid = List[List[int]]


def dims(g: Grid) -> Tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def rotate90(g: Grid) -> Grid:
    h, w = dims(g)
    return [[g[h - 1 - r][c] for r in range(h)] for c in range(w)]


def rotate180(g: Grid) -> Grid:
    return [row[::-1] for row in g[::-1]]


def rotate270(g: Grid) -> Grid:
    h, w = dims(g)
    return [[g[r][w - 1 - c] for r in range(h)] for c in range(w - 1, -1, -1)]


def flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]


def flip_v(g: Grid) -> Grid:
    return g[::-1]


def transpose(g: Grid) -> Grid:
    h, w = dims(g)
    return [[g[r][c] for r in range(h)] for c in range(w)]


Op = str  # e.g., 'rot90', 'rot180', 'rot270', 'flip_h', 'flip_v', 'transpose'


def apply_geom(g: Grid, ops: List[Op]) -> Grid:
    out = g
    for op in ops:
        if op == "rot90":
            out = rotate90(out)
        elif op == "rot180":
            out = rotate180(out)
        elif op == "rot270":
            out = rotate270(out)
        elif op == "flip_h":
            out = flip_h(out)
        elif op == "flip_v":
            out = flip_v(out)
        elif op == "transpose":
            out = transpose(out)
        else:
            # Unknown op: ignore (should not happen in controlled enumeration)
            pass
    return out


# -----------------
# Palette Mapping
# -----------------

PaletteMap = Dict[int, int]


def infer_bijective_palette_map(src: Grid, dst: Grid) -> Optional[PaletteMap]:
    """Infer a 1-1 color mapping such that applying it to src yields dst.
    Returns None if inconsistent. Only constrains seen colors; unseen map to themselves.
    """
    if dims(src) != dims(dst):
        return None
    mapping: Dict[int, int] = {}
    inverse: Dict[int, int] = {}
    h, w = dims(src)
    for r in range(h):
        for c in range(w):
            s, d = src[r][c], dst[r][c]
            if s in mapping and mapping[s] != d:
                return None
            if d in inverse and inverse[d] != s:
                return None
            mapping[s] = d
            inverse[d] = s
    return mapping


def merge_palette_maps(global_map: PaletteMap, local_map: PaletteMap) -> Optional[PaletteMap]:
    merged = dict(global_map)
    inv: Dict[int, int] = {v: k for k, v in merged.items()}
    for k, v in local_map.items():
        if k in merged and merged[k] != v:
            return None
        if v in inv and inv[v] != k:
            return None
        merged[k] = v
        inv[v] = k
    return merged


def apply_palette(g: Grid, mapping: PaletteMap) -> Grid:
    return [[mapping.get(v, v) for v in row] for row in g]


# -----------------
# Training Fit Checkers
# -----------------

def fit_palette_only(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[PaletteMap]:
    """Check if a single global palette relabel maps inputs to outputs across all pairs."""
    global_map: PaletteMap = {}
    for src, dst in train_pairs:
        local = infer_bijective_palette_map(src, dst)
        if local is None:
            return None
        merged = merge_palette_maps(global_map, local)
        if merged is None:
            return None
        global_map = merged
    return global_map


def fit_geom_only(train_pairs: List[Tuple[Grid, Grid]], ops: List[Op]) -> bool:
    for src, dst in train_pairs:
        trans = apply_geom(src, ops)
        if trans != dst:
            return False
    return True


def fit_geom_plus_palette(train_pairs: List[Tuple[Grid, Grid]], ops: List[Op]) -> Optional[PaletteMap]:
    global_map: PaletteMap = {}
    for src, dst in train_pairs:
        trans = apply_geom(src, ops)
        local = infer_bijective_palette_map(trans, dst)
        if local is None:
            return None
        merged = merge_palette_maps(global_map, local)
        if merged is None:
            return None
        global_map = merged
    return global_map


# -----------------
# Candidate Enumeration (V1)
# -----------------

BASE_OPS: List[Op] = ["rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"]


def enumerate_geom_sequences() -> List[List[Op]]:
    seqs: List[List[Op]] = []
    # length 1
    for op in BASE_OPS:
        seqs.append([op])
    # length 2
    for i, op1 in enumerate(BASE_OPS):
        for op2 in BASE_OPS:
            seqs.append([op1, op2])
    return seqs


# -----------------
# Solver per task
# -----------------

def apply_program(g: Grid, ops: List[Op], pmap: Optional[PaletteMap]) -> Grid:
    out = apply_geom(g, ops)
    if pmap is not None:
        out = apply_palette(out, pmap)
    return out


def solve_task_v1(task: Dict[str, Any]) -> Tuple[List[Dict[str, Grid]], Dict[str, Any]]:
    """Return two attempts per test input and a debug info dict.
    Attempts are built via two tracks: palette-first and geometry-first.
    """
    train = task.get("train", [])
    test = task.get("test", [])
    train_pairs: List[Tuple[Grid, Grid]] = [(t["input"], t["output"]) for t in train]
    geom_seqs = enumerate_geom_sequences()

    found_prog_attempt1: Optional[Tuple[List[Op], Optional[PaletteMap]]] = None
    found_prog_attempt2: Optional[Tuple[List[Op], Optional[PaletteMap]]] = None

    # Attempt 1: palette-first
    pmap = fit_palette_only(train_pairs)
    if pmap is not None:
        found_prog_attempt1 = ([], pmap)
    if found_prog_attempt1 is None:
        for ops in geom_seqs:
            pmap2 = fit_geom_plus_palette(train_pairs, ops)
            if pmap2 is not None:
                found_prog_attempt1 = (ops, pmap2)
                break

    # Attempt 2: geometry-first
    for ops in geom_seqs:
        if fit_geom_only(train_pairs, ops):
            # Avoid duplicate program if identical to attempt_1
            if not (found_prog_attempt1 is not None and found_prog_attempt1[0] == ops and found_prog_attempt1[1] is None):
                found_prog_attempt2 = (ops, None)
                break
    if found_prog_attempt2 is None:
        for ops in geom_seqs:
            pmap3 = fit_geom_plus_palette(train_pairs, ops)
            if pmap3 is not None:
                # Avoid exact duplicate of attempt_1
                if not (found_prog_attempt1 is not None and found_prog_attempt1[0] == ops and found_prog_attempt1[1] == pmap3):
                    found_prog_attempt2 = (ops, pmap3)
                    break

    # Build predictions for test inputs
    preds: List[Dict[str, Grid]] = []
    for item in test:
        tin: Grid = item.get("input", [])
        # attempt_1
        if found_prog_attempt1 is not None:
            ops1, map1 = found_prog_attempt1
            a1 = apply_program(tin, ops1, map1)
        else:
            a1 = tin
        # attempt_2
        if found_prog_attempt2 is not None:
            ops2, map2 = found_prog_attempt2
            a2 = apply_program(tin, ops2, map2)
        else:
            a2 = tin
        preds.append({"attempt_1": a1, "attempt_2": a2})

    debug = {
        "attempt_1": found_prog_attempt1,
        "attempt_2": found_prog_attempt2,
    }
    return preds, debug


# -----------------
# Evaluation Harness
# -----------------

def eval_exact_attempts(gt: List[Grid], pred: List[Dict[str, Grid]]) -> float:
    solved = 0
    for y, yhat in zip(gt, pred):
        if yhat.get("attempt_1") == y or yhat.get("attempt_2") == y:
            solved += 1
    return solved / max(1, len(gt))


def evaluate_on_split(challenges: Dict[str, Any], solutions: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    task_scores: Dict[str, float] = {}
    total_correct = 0
    total_items = 0
    for tid, task in challenges.items():
        preds, _ = solve_task_v1(task)
        gt_list: List[Grid] = solutions.get(tid, [])
        # Ensure lengths match; if not, truncate to common length
        k = min(len(gt_list), len(preds))
        acc = eval_exact_attempts(gt_list[:k], preds[:k])
        task_scores[tid] = acc
        total_correct += int(acc * k)
        total_items += k
    overall = (total_correct / total_items) if total_items > 0 else 0.0
    return overall, task_scores


# -----------------
# CLI
# -----------------

def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in {"eval", "submit"}:
        print(
            "Usage:\n"
            "  python arc_prize_2025_submission/versions/arc_solver_v1.py eval <evaluation_challenges_json> <evaluation_solutions_json>\n"
            "  python arc_prize_2025_submission/versions/arc_solver_v1.py submit <test_challenges_json> <output_submission_json>"
        )
        sys.exit(1)

    mode = args[0]
    if mode == "eval":
        if len(args) != 3:
            print("eval mode requires <evaluation_challenges_json> <evaluation_solutions_json>")
            sys.exit(1)
        chall_path, sol_path = args[1], args[2]
        challenges = load_json(chall_path)
        solutions = load_json(sol_path)
        overall, task_scores = evaluate_on_split(challenges, solutions)
        print(f"Overall accuracy: {overall:.4f}")
        # Print a few sample task scores deterministically
        for tid in list(sorted(task_scores.keys()))[:10]:
            print(f"  {tid}: {task_scores[tid]:.4f}")
    else:  # submit
        if len(args) != 3:
            print("submit mode requires <test_challenges_json> <output_submission_json>")
            sys.exit(1)
        chall_path, out_path = args[1], args[2]
        challenges = load_json(chall_path)
        submission: Dict[str, Any] = {}
        for tid, task in challenges.items():
            preds, _ = solve_task_v1(task)
            submission[tid] = preds
        write_json(out_path, submission)
        print(f"Wrote submission to {out_path}")


if __name__ == "__main__":
    main()
