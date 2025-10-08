"""
ARC Solver V2 (expanded DSL + guided search)

Extensions over V1:
- Adds parameterized ops: tile(hr, wr), pad_crop(dh, dw, fill).
- Feature-driven parameterization (ratios, deltas, fill color).
- Expands candidate enumeration with geometry, tile, pad/crop.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple, Optional


Grid = List[List[int]]
Op = Tuple[str, tuple]


# -----------------
# IO Utilities (optional)
# -----------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


# -----------------
# Basic Grid Helpers
# -----------------

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


def tile_scale(g: Grid, hr: int, wr: int) -> Grid:
    if hr <= 0 or wr <= 0:
        return g
    out: Grid = []
    for row in g:
        scaled_row = []
        for v in row:
            scaled_row.extend([v] * wr)
        for _ in range(hr):
            out.append(list(scaled_row))
    return out


def pad_crop_center(g: Grid, dh: int, dw: int, fill: int) -> Grid:
    h, w = dims(g)
    th, tw = max(1, h + dh), max(1, w + dw)
    # Start from all fill grid
    out: Grid = [[fill for _ in range(tw)] for _ in range(th)]
    # Determine source region to copy (centered crop/pad)
    src_top = 0
    src_left = 0
    copy_h = h
    copy_w = w
    # If cropping, adjust source region
    if h > th:
        excess = h - th
        src_top = excess // 2
        copy_h = th
    if w > tw:
        excess = w - tw
        src_left = excess // 2
        copy_w = tw
    # Destination offset (centered)
    dst_top = max(0, (th - copy_h) // 2)
    dst_left = max(0, (tw - copy_w) // 2)
    for r in range(copy_h):
        for c in range(copy_w):
            out[dst_top + r][dst_left + c] = g[src_top + r][src_left + c]
    return out


def translate(g: Grid, dy: int, dx: int, fill: int) -> Grid:
    h, w = dims(g)
    out: Grid = [[fill for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            nr, nc = r + dy, c + dx
            if 0 <= nr < h and 0 <= nc < w:
                out[nr][nc] = g[r][c]
    return out


# -----------------
# Palette Mapping (same logic as V1)
# -----------------

PaletteMap = Dict[int, int]


def infer_bijective_palette_map(src: Grid, dst: Grid) -> Optional[PaletteMap]:
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
# Feature Extraction
# -----------------

def most_common_color(g: Grid) -> int:
    freq: Dict[int, int] = {}
    for row in g:
        for v in row:
            freq[v] = freq.get(v, 0) + 1
    best = 0
    best_c = 0
    for c, f in freq.items():
        if f > best:
            best = f
            best_c = c
    return best_c


def infer_size_ratios(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Tuple[int, int]]:
    hr: Optional[int] = None
    wr: Optional[int] = None
    for src, dst in train_pairs:
        h1, w1 = dims(src)
        h2, w2 = dims(dst)
        if h1 == 0 or w1 == 0:
            return None
        if h2 % h1 != 0 or w2 % w1 != 0:
            return None
        r_h, r_w = h2 // h1, w2 // w1
        if hr is None:
            hr = r_h
            wr = r_w
        else:
            if hr != r_h or wr != r_w:
                return None
    return (hr or 1, wr or 1)


def infer_size_deltas(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Tuple[int, int]]:
    dh: Optional[int] = None
    dw: Optional[int] = None
    for src, dst in train_pairs:
        h1, w1 = dims(src)
        h2, w2 = dims(dst)
        d_h, d_w = h2 - h1, w2 - w1
        if dh is None:
            dh, dw = d_h, d_w
        else:
            if dh != d_h or dw != d_w:
                return None
    return (dh or 0, dw or 0)


# -----------------
# Operation Application
# -----------------

def apply_ops(g: Grid, ops: List[Op]) -> Grid:
    out = g
    for kind, params in ops:
        if kind == "geom":
            op = params[0]
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
        elif kind == "tile":
            hr, wr = params
            out = tile_scale(out, hr, wr)
        elif kind == "pad_crop":
            dh, dw, fill = params
            out = pad_crop_center(out, dh, dw, fill)
        elif kind == "translate":
            dy, dx, fill = params
            out = translate(out, dy, dx, fill)
        else:
            # Unknown op: ignore
            pass
    return out


# -----------------
# Fitting Programs
# -----------------

def fit_palette_only(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[PaletteMap]:
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


def fit_program(train_pairs: List[Tuple[Grid, Grid]], ops: List[Op], allow_palette: bool) -> Optional[Tuple[List[Op], Optional[PaletteMap]]]:
    if not allow_palette:
        for src, dst in train_pairs:
            out = apply_ops(src, ops)
            if out != dst:
                return None
        return (ops, None)
    else:
        global_map: PaletteMap = {}
        for src, dst in train_pairs:
            out = apply_ops(src, ops)
            local = infer_bijective_palette_map(out, dst)
            if local is None:
                return None
            merged = merge_palette_maps(global_map, local)
            if merged is None:
                return None
            global_map = merged
        return (ops, global_map)


# -----------------
# Candidate Enumeration (feature-driven)
# -----------------

BASE_GEOMS = ["rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"]


def enumerate_candidates(train_pairs: List[Tuple[Grid, Grid]]) -> List[List[Op]]:
    cands: List[List[Op]] = []
    # Geom only: length 1 and 2
    for g in BASE_GEOMS:
        cands.append([("geom", (g,))])
    for g1 in BASE_GEOMS:
        for g2 in BASE_GEOMS:
            cands.append([("geom", (g1,)), ("geom", (g2,))])

    # Feature-driven: tile and pad/crop
    ratios = infer_size_ratios(train_pairs)
    deltas = infer_size_deltas(train_pairs)

    # Fill color: most common across all train outputs
    fill_votes: Dict[int, int] = {}
    for _, dst in train_pairs:
        c = most_common_color(dst)
        fill_votes[c] = fill_votes.get(c, 0) + 1
    fill = (max(fill_votes.items(), key=lambda kv: kv[1])[0]) if fill_votes else 0

    if ratios is not None:
        hr, wr = ratios
        if not (hr == 1 and wr == 1):
            # tile alone and with one geom
            cands.append([("tile", (hr, wr))])
            for g in BASE_GEOMS:
                cands.append([("tile", (hr, wr)), ("geom", (g,))])
                cands.append([("geom", (g,)), ("tile", (hr, wr))])

    if deltas is not None:
        dh, dw = deltas
        if not (dh == 0 and dw == 0):
            # pad/crop alone and with one geom
            cands.append([("pad_crop", (dh, dw, fill))])
            for g in BASE_GEOMS:
                cands.append([("pad_crop", (dh, dw, fill)), ("geom", (g,))])
                cands.append([("geom", (g,)), ("pad_crop", (dh, dw, fill))])

    # Translation: search small window and require consistent (dy,dx) across pairs (dims equal)
    # Limit search to |dy|,|dx| <= 3 to keep it cheap.
    h_eq = all(dims(src) == dims(dst) for src, dst in train_pairs)
    if h_eq:
        possible: Optional[Tuple[int, int]] = None
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ok = True
                for src, dst in train_pairs:
                    # Use fill from dst mode to align edges
                    trial = translate(src, dy, dx, most_common_color(dst))
                    if trial != dst:
                        ok = False
                        break
                if ok:
                    possible = (dy, dx)
                    break
            if possible is not None:
                break
        if possible is not None:
            dy, dx = possible
            cands.append([("translate", (dy, dx, fill))])
            for g in BASE_GEOMS:
                cands.append([("translate", (dy, dx, fill)), ("geom", (g,))])
                cands.append([("geom", (g,)), ("translate", (dy, dx, fill))])

    # De-duplicate candidates by a simple string key
    seen = set()
    uniq: List[List[Op]] = []
    for seq in cands:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            uniq.append(seq)
    return uniq


# -----------------
# Solve task (V2)
# -----------------

def apply_program(g: Grid, ops: List[Op], pmap: Optional[PaletteMap]) -> Grid:
    out = apply_ops(g, ops)
    if pmap is not None:
        out = apply_palette(out, pmap)
    return out


def solve_task_v2(task: Dict[str, Any]) -> Tuple[List[Dict[str, Grid]], Dict[str, Any]]:
    train = task.get("train", [])
    test = task.get("test", [])
    train_pairs: List[Tuple[Grid, Grid]] = [(t["input"], t["output"]) for t in train]

    candidates = enumerate_candidates(train_pairs)

    attempt1: Optional[Tuple[List[Op], Optional[PaletteMap]]] = None
    attempt2: Optional[Tuple[List[Op], Optional[PaletteMap]]] = None

    # Attempt 1: palette-first
    pmap = fit_palette_only(train_pairs)
    if pmap is not None:
        attempt1 = ([], pmap)
    if attempt1 is None:
        # Try candidates with palette allowed
        for seq in candidates:
            fit = fit_program(train_pairs, seq, allow_palette=True)
            if fit is not None:
                attempt1 = fit
                break

    # Attempt 2: geometry-first (prefer exact without palette)
    for seq in candidates:
        fit = fit_program(train_pairs, seq, allow_palette=False)
        if fit is not None:
            if not (attempt1 is not None and attempt1[0] == fit[0] and attempt1[1] == fit[1]):
                attempt2 = fit
                break
    if attempt2 is None:
        for seq in candidates:
            fit = fit_program(train_pairs, seq, allow_palette=True)
            if fit is not None:
                if not (attempt1 is not None and attempt1[0] == fit[0] and attempt1[1] == fit[1]):
                    attempt2 = fit
                    break

    # Build predictions
    preds: List[Dict[str, Grid]] = []
    for item in test:
        tin: Grid = item.get("input", [])
        if attempt1 is not None:
            a1 = apply_program(tin, attempt1[0], attempt1[1])
        else:
            a1 = tin
        if attempt2 is not None:
            a2 = apply_program(tin, attempt2[0], attempt2[1])
        else:
            a2 = tin
        preds.append({"attempt_1": a1, "attempt_2": a2})

    debug = {"attempt_1": attempt1, "attempt_2": attempt2}
    return preds, debug


# -----------------
# Evaluation Harness and CLI (optional, parity with V1)
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
        preds, _ = solve_task_v2(task)
        gt_list: List[Grid] = solutions.get(tid, [])
        k = min(len(gt_list), len(preds))
        acc = eval_exact_attempts(gt_list[:k], preds[:k])
        task_scores[tid] = acc
        total_correct += int(acc * k)
        total_items += k
    overall = (total_correct / total_items) if total_items > 0 else 0.0
    return overall, task_scores


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args or args[0] not in {"eval", "submit"}:
        print(
            "Usage:\n"
            "  python arc_prize_2025_submission/versions/arc_solver_v2.py eval <evaluation_challenges_json> <evaluation_solutions_json>\n"
            "  python arc_prize_2025_submission/versions/arc_solver_v2.py submit <test_challenges_json> <output_submission_json>"
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
        print(f"Overall accuracy (V2): {overall:.4f}")
        for tid in list(sorted(task_scores.keys()))[:10]:
            print(f"  {tid}: {task_scores[tid]:.4f}")
    else:
        if len(args) != 3:
            print("submit mode requires <test_challenges_json> <output_submission_json>")
            sys.exit(1)
        chall_path, out_path = args[1], args[2]
        challenges = load_json(chall_path)
        submission: Dict[str, Any] = {}
        for tid, task in challenges.items():
            preds, _ = solve_task_v2(task)
            submission[tid] = preds
        write_json(out_path, submission)
        print(f"Wrote submission (V2) to {out_path}")
