"""
ARC Solver V3 (components + masks + expanded candidates)

Adds connected-component operations and component-driven heuristics to V2:
- Connected components (4-neighborhood) per color.
- Largest non-background component crop and keep operations.
- Reuses V2 ops: geometry, tile, pad/crop, translate, palette mapping.
- Evaluation harness + CLI retained for parity.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple, Optional, Deque
from collections import deque


Grid = List[List[int]]
Op = Tuple[str, tuple]


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
    out: Grid = [[fill for _ in range(tw)] for _ in range(th)]
    src_top = 0
    src_left = 0
    copy_h = h
    copy_w = w
    if h > th:
        excess = h - th
        src_top = excess // 2
        copy_h = th
    if w > tw:
        excess = w - tw
        src_left = excess // 2
        copy_w = tw
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


def add_border(g: Grid, color: int) -> Grid:
    h, w = dims(g)
    out: Grid = [[color for _ in range(w + 2)] for _ in range(h + 2)]
    for r in range(h):
        for c in range(w):
            out[r + 1][c + 1] = g[r][c]
    return out


def remove_border(g: Grid) -> Optional[Grid]:
    """Remove a 1-cell border if all border cells share the same color."""
    h, w = dims(g)
    if h <= 2 or w <= 2:
        return None
    # Check constant border color
    border_vals = []
    for c in range(w):
        border_vals.append(g[0][c])
        border_vals.append(g[h - 1][c])
    for r in range(h):
        border_vals.append(g[r][0])
        border_vals.append(g[r][w - 1])
    if len(set(border_vals)) != 1:
        return None
    # Remove border
    out: Grid = []
    for r in range(1, h - 1):
        out.append(g[r][1:w - 1])
    return out


def symmetrize_h_left_to_right(g: Grid) -> Grid:
    h, w = dims(g)
    out: Grid = [row[:] for row in g]
    for r in range(h):
        for c in range(w // 2, w):
            out[r][c] = out[r][w - 1 - c]
    return out


def symmetrize_v_top_to_bottom(g: Grid) -> Grid:
    h, w = dims(g)
    out: Grid = [row[:] for row in g]
    for r in range(h // 2, h):
        out[r] = out[h - 1 - r][:]
    return out


# -----------------
# Palette Mapping
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


def infer_palette_map_surjective(src: Grid, dst: Grid) -> Optional[PaletteMap]:
    """Infer a many-to-one palette map (surjective to seen dst colors in src region).
    Only requires that a given src color maps consistently to a single dst color.
    Different src colors may map to the same dst color. Dimensions must match.
    """
    if dims(src) != dims(dst):
        return None
    mapping: Dict[int, int] = {}
    h, w = dims(src)
    for r in range(h):
        for c in range(w):
            s, d = src[r][c], dst[r][c]
            if s in mapping and mapping[s] != d:
                return None
            mapping[s] = d
    return mapping


def merge_palette_maps_surjective(global_map: PaletteMap, local_map: PaletteMap) -> Optional[PaletteMap]:
    merged = dict(global_map)
    for k, v in local_map.items():
        if k in merged and merged[k] != v:
            return None
        merged[k] = v
    return merged


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
# Connected Components (4-neighborhood per color)
# -----------------

def cc_label_color(g: Grid, target_color: int) -> List[List[int]]:
    h, w = dims(g)
    labels = [[-1 for _ in range(w)] for _ in range(h)]
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if g[r][c] == target_color and labels[r][c] == -1:
                # BFS
                q: Deque[Tuple[int, int]] = deque()
                q.append((r, c))
                labels[r][c] = comp_id
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and labels[nr][nc] == -1 and g[nr][nc] == target_color:
                            labels[nr][nc] = comp_id
                            q.append((nr, nc))
                comp_id += 1
    return labels


def largest_nonbg_component_bbox(g: Grid) -> Optional[Tuple[int, int, int, int]]:
    h, w = dims(g)
    bg = most_common_color(g)
    best_size = 0
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    # Consider each color except bg and measure largest CC size and bbox
    colors: Dict[int, bool] = {}
    for r in range(h):
        for c in range(w):
            colors[g[r][c]] = True
    for color in colors.keys():
        if color == bg:
            continue
        labels = cc_label_color(g, color)
        # collect sizes and bboxes
        sizes: Dict[int, int] = {}
        bboxes: Dict[int, Tuple[int, int, int, int]] = {}
        for r in range(h):
            for c in range(w):
                cid = labels[r][c]
                if cid >= 0:
                    sizes[cid] = sizes.get(cid, 0) + 1
                    if cid not in bboxes:
                        bboxes[cid] = (r, c, r, c)
                    else:
                        r0, c0, r1, c1 = bboxes[cid]
                        bboxes[cid] = (min(r0, r), min(c0, c), max(r1, r), max(c1, c))
        for cid, sz in sizes.items():
            if sz > best_size:
                best_size = sz
                best_bbox = bboxes[cid]
    return best_bbox


def crop_bbox(g: Grid, bbox: Tuple[int, int, int, int]) -> Grid:
    r0, c0, r1, c1 = bbox
    out: Grid = []
    for r in range(r0, r1 + 1):
        out.append(g[r][c0:c1 + 1])
    return out


def keep_largest_nonbg(g: Grid, fill: int) -> Grid:
    h, w = dims(g)
    bg = most_common_color(g)
    # Find largest non-bg component by size across colors
    bbox = largest_nonbg_component_bbox(g)
    if bbox is None:
        return [[fill for _ in range(w)] for _ in range(h)]
    r0, c0, r1, c1 = bbox
    out = [[fill for _ in range(w)] for _ in range(h)]
    # Fill only within bbox where colors are not bg and connected to that bbox's component(s)
    # Simpler: keep all non-bg pixels inside bbox; conservative but useful for many tasks
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if g[r][c] != bg:
                out[r][c] = g[r][c]
    return out


def move_largest_nonbg_to_topleft(g: Grid, fill: int) -> Grid:
    bbox = largest_nonbg_component_bbox(g)
    if bbox is None:
        h, w = dims(g)
        return [[fill for _ in range(w)] for _ in range(h)]
    cropped = crop_bbox(g, bbox)
    ch, cw = dims(cropped)
    h, w = dims(g)
    out: Grid = [[fill for _ in range(w)] for _ in range(h)]
    # place at (0,0)
    for r in range(min(ch, h)):
        for c in range(min(cw, w)):
            out[r][c] = cropped[r][c]
    return out


def extract_largest_nonbg(g: Grid) -> Optional[Tuple[Grid, Tuple[int, int]]]:
    """Return (cropped_component_grid, (tl_r, tl_c)) for the largest non-bg component."""
    bbox = largest_nonbg_component_bbox(g)
    if bbox is None:
        return None
    r0, c0, r1, c1 = bbox
    return crop_bbox(g, bbox), (r0, c0)


def move_largest_nonbg_by(g: Grid, dy: int, dx: int, fill: int) -> Grid:
    h, w = dims(g)
    ext = extract_largest_nonbg(g)
    if ext is None:
        return [[fill for _ in range(w)] for _ in range(h)]
    comp, (r0, c0) = ext
    ch, cw = dims(comp)
    out: Grid = [[fill for _ in range(w)] for _ in range(h)]
    dst_r0 = r0 + dy
    dst_c0 = c0 + dx
    for r in range(ch):
        for c in range(cw):
            nr = dst_r0 + r
            nc = dst_c0 + c
            if 0 <= nr < h and 0 <= nc < w:
                out[nr][nc] = comp[r][c]
    return out


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
        elif kind == "crop_largest_nonbg":
            bbox = largest_nonbg_component_bbox(out)
            if bbox is not None:
                out = crop_bbox(out, bbox)
        elif kind == "keep_largest_nonbg":
            fill = params[0]
            out = keep_largest_nonbg(out, fill)
        elif kind == "move_largest_nonbg_to_tl":
            fill = params[0]
            out = move_largest_nonbg_to_topleft(out, fill)
        elif kind == "remove_border":
            tmp = remove_border(out)
            if tmp is not None:
                out = tmp
        elif kind == "add_border":
            color = params[0]
            out = add_border(out, color)
        elif kind == "sym_h_lr":
            out = symmetrize_h_left_to_right(out)
        elif kind == "sym_v_tb":
            out = symmetrize_v_top_to_bottom(out)
        elif kind == "move_largest_nonbg_by":
            dy, dx, fill = params
            out = move_largest_nonbg_by(out, dy, dx, fill)
        else:
            pass
    return out


# -----------------
# Fitting Programs
# -----------------

PaletteMap = Dict[int, int]


def infer_bijective_palette_map(src: Grid, dst: Grid) -> Optional[PaletteMap]:
    # already defined above; duplicating name for clarity in this scope
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


def apply_palette_map(g: Grid, mapping: PaletteMap) -> Grid:
    return [[mapping.get(v, v) for v in row] for row in g]


def fit_program(train_pairs: List[Tuple[Grid, Grid]], ops: List[Op], allow_palette: bool) -> Optional[Tuple[List[Op], Optional[PaletteMap]]]:
    if not allow_palette:
        for src, dst in train_pairs:
            out = apply_ops(src, ops)
            if out != dst:
                return None
        return (ops, None)
    else:
        # Try bijective first
        global_map: PaletteMap = {}
        ok = True
        for src, dst in train_pairs:
            out = apply_ops(src, ops)
            local = infer_bijective_palette_map(out, dst)
            if local is None:
                ok = False
                break
            merged = merge_palette_maps(global_map, local)
            if merged is None:
                ok = False
                break
            global_map = merged
        if ok:
            return (ops, global_map)
        # Fallback to surjective (many-to-one) mapping
        global_map2: PaletteMap = {}
        for src, dst in train_pairs:
            out = apply_ops(src, ops)
            local2 = infer_palette_map_surjective(out, dst)
            if local2 is None:
                return None
            merged2 = merge_palette_maps_surjective(global_map2, local2)
            if merged2 is None:
                return None
            global_map2 = merged2
        return (ops, global_map2)


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


# -----------------
# Candidate Enumeration (feature-driven + components)
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
    fill_votes: Dict[int, int] = {}
    for _, dst in train_pairs:
        c = most_common_color(dst)
        fill_votes[c] = fill_votes.get(c, 0) + 1
    fill = (max(fill_votes.items(), key=lambda kv: kv[1])[0]) if fill_votes else 0

    if ratios is not None:
        hr, wr = ratios
        if not (hr == 1 and wr == 1):
            cands.append([("tile", (hr, wr))])
            for g in BASE_GEOMS:
                cands.append([("tile", (hr, wr)), ("geom", (g,))])
                cands.append([("geom", (g,)), ("tile", (hr, wr))])

    if deltas is not None:
        dh, dw = deltas
        if not (dh == 0 and dw == 0):
            cands.append([("pad_crop", (dh, dw, fill))])
            for g in BASE_GEOMS:
                cands.append([("pad_crop", (dh, dw, fill)), ("geom", (g,))])
                cands.append([("geom", (g,)), ("pad_crop", (dh, dw, fill))])

    # Translation: small window if dims equal across pairs
    same_dims = all(dims(src) == dims(dst) for src, dst in train_pairs)
    if same_dims:
        possible: Optional[Tuple[int, int]] = None
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ok = True
                for src, dst in train_pairs:
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

    # Component-driven candidates: crop/keep largest non-bg
    # Include if outputs tend to be smaller or sparse
    suggest_crop = any(dims(dst)[0] <= dims(src)[0] and dims(dst)[1] <= dims(src)[1] for src, dst in train_pairs)
    if suggest_crop:
        cands.append([("crop_largest_nonbg", tuple())])
        for g in BASE_GEOMS:
            cands.append([("crop_largest_nonbg", tuple()), ("geom", (g,))])
            cands.append([("geom", (g,)), ("crop_largest_nonbg", tuple())])
        # keep largest non-bg with fill
        cands.append([("keep_largest_nonbg", (fill,))])
        for g in BASE_GEOMS:
            cands.append([("keep_largest_nonbg", (fill,)), ("geom", (g,))])
            cands.append([("geom", (g,)), ("keep_largest_nonbg", (fill,))])

    # Border ops if consistent deltas of -2
    if deltas is not None and deltas == (-2, -2):
        cands.append([("remove_border", tuple())])
        for g in BASE_GEOMS:
            cands.append([("remove_border", tuple()), ("geom", (g,))])
            cands.append([("geom", (g,)), ("remove_border", tuple())])
    if deltas is not None and deltas == (2, 2):
        cands.append([("add_border", (fill,))])
        for g in BASE_GEOMS:
            cands.append([("add_border", (fill,)), ("geom", (g,))])
            cands.append([("geom", (g,)), ("add_border", (fill,))])

    # Move largest component to top-left (useful when output is aligned to origin)
    cands.append([("move_largest_nonbg_to_tl", (fill,))])

    # Symmetry enforcement (limited)
    cands.append([("sym_h_lr", tuple())])
    cands.append([("sym_v_tb", tuple())])

    # Component translation by consistent delta across pairs (heuristic)
    # Compute delta between largest component TLs if consistent
    deltas_cc: Optional[Tuple[int, int]] = None
    ok = True
    for src, dst in train_pairs:
        es = extract_largest_nonbg(src)
        ed = extract_largest_nonbg(dst)
        if es is None or ed is None:
            ok = False
            break
        (_, (rs, cs)) = es
        (_, (rd, cd)) = ed
        dyy, dxx = (rd - rs, cd - cs)
        if deltas_cc is None:
            deltas_cc = (dyy, dxx)
        else:
            if deltas_cc != (dyy, dxx):
                ok = False
                break
    if ok and deltas_cc is not None:
        dy, dx = deltas_cc
        cands.append([("move_largest_nonbg_by", (dy, dx, fill))])

    # De-duplicate
    seen = set()
    uniq: List[List[Op]] = []
    for seq in cands:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            uniq.append(seq)
    # Canonicalize geometry duplicates via signature on a small probe grid
    def signature(ops: List[Op]) -> Tuple:
        probe: Grid = [[0, 1, 2], [3, 4, 5]]
        out = apply_ops(probe, [op for op in ops if op[0] == "geom"])  # only geom affects signature
        return tuple(tuple(row) for row in out), len([op for op in ops if op[0] != "geom"])  # shape + count non-geom
    seen_sig = set()
    final: List[List[Op]] = []
    for seq in uniq:
        sig = signature(seq)
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        final.append(seq)
    return final


def build_operator_pool(train_pairs: List[Tuple[Grid, Grid]]) -> List[Op]:
    pool: List[Op] = []
    # Geom atoms
    for g in BASE_GEOMS:
        pool.append(("geom", (g,)))
    # Feature-driven params
    ratios = infer_size_ratios(train_pairs)
    deltas = infer_size_deltas(train_pairs)
    fill_votes: Dict[int, int] = {}
    for _, dst in train_pairs:
        c = most_common_color(dst)
        fill_votes[c] = fill_votes.get(c, 0) + 1
    fill = (max(fill_votes.items(), key=lambda kv: kv[1])[0]) if fill_votes else 0
    if ratios is not None:
        hr, wr = ratios
        if not (hr == 1 and wr == 1):
            pool.append(("tile", (hr, wr)))
    if deltas is not None:
        dh, dw = deltas
        if not (dh == 0 and dw == 0):
            pool.append(("pad_crop", (dh, dw, fill)))
        if deltas == (-2, -2):
            pool.append(("remove_border", tuple()))
        if deltas == (2, 2):
            pool.append(("add_border", (fill,)))
    # Translation: only include if a small consistent offset found
    same_dims = all(dims(src) == dims(dst) for src, dst in train_pairs)
    if same_dims:
        found: Optional[Tuple[int, int]] = None
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ok = True
                for src, dst in train_pairs:
                    trial = translate(src, dy, dx, most_common_color(dst))
                    if trial != dst:
                        ok = False
                        break
                if ok:
                    found = (dy, dx)
                    break
            if found is not None:
                break
        if found is not None:
            dy, dx = found
            pool.append(("translate", (dy, dx, fill)))
    # Component ops and symmetry
    pool.append(("crop_largest_nonbg", tuple()))
    pool.append(("keep_largest_nonbg", (fill,)))
    pool.append(("move_largest_nonbg_to_tl", (fill,)))
    pool.append(("sym_h_lr", tuple()))
    pool.append(("sym_v_tb", tuple()))
    # De-duplicate
    seen = set()
    uniq: List[Op] = []
    for op in pool:
        if op not in seen:
            seen.add(op)
            uniq.append(op)
    return uniq


def beam_search_fit(train_pairs: List[Tuple[Grid, Grid]], pool: List[Op], beam: int = 12, depth: int = 3, allow_palette: bool = True) -> Optional[Tuple[List[Op], Optional[PaletteMap]]]:
    # Nodes are (ops_seq)
    frontier: List[List[Op]] = [[]]
    tried = set()
    for d in range(1, depth + 1):
        new_frontier: List[List[Op]] = []
        for seq in frontier:
            for op in pool:
                new_seq = seq + [op]
                key = tuple(new_seq)
                if key in tried:
                    continue
                tried.add(key)
                fit = fit_program(train_pairs, new_seq, allow_palette=allow_palette)
                if fit is not None:
                    return fit
                # Keep as candidate for next layer; simple heuristic: keep first N by length (FIFO order)
                new_frontier.append(new_seq)
                if len(new_frontier) > beam:
                    new_frontier = new_frontier[:beam]
        frontier = new_frontier
        if not frontier:
            break
    return None


# -----------------
# Solve task (V3)
# -----------------

def apply_program(g: Grid, ops: List[Op], pmap: Optional[PaletteMap]) -> Grid:
    out = apply_ops(g, ops)
    if pmap is not None:
        out = apply_palette_map(out, pmap)
    return out


def solve_task_v3(task: Dict[str, Any]) -> Tuple[List[Dict[str, Grid]], Dict[str, Any]]:
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
        for seq in candidates:
            fit = fit_program(train_pairs, seq, allow_palette=True)
            if fit is not None:
                attempt1 = fit
                break
    if attempt1 is None:
        # Beam search with palette allowed
        pool = build_operator_pool(train_pairs)
        fit = beam_search_fit(train_pairs, pool, beam=12, depth=3, allow_palette=True)
        if fit is not None:
            attempt1 = fit

    # Attempt 2: geometry/components-first without palette, then with palette
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
    if attempt2 is None:
        # Beam search without palette first
        pool = build_operator_pool(train_pairs)
        fit = beam_search_fit(train_pairs, pool, beam=12, depth=3, allow_palette=False)
        if fit is None:
            fit = beam_search_fit(train_pairs, pool, beam=12, depth=3, allow_palette=True)
        if fit is not None:
            if not (attempt1 is not None and attempt1[0] == fit[0] and attempt1[1] == fit[1]):
                attempt2 = fit

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
# Evaluation Harness and CLI
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
        preds, _ = solve_task_v3(task)
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
            "  python arc_prize_2025_submission/versions/arc_solver_v3.py eval <evaluation_challenges_json> <evaluation_solutions_json>\n"
            "  python arc_prize_2025_submission/versions/arc_solver_v3.py submit <test_challenges_json> <output_submission_json>"
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
        print(f"Overall accuracy (V3): {overall:.4f}")
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
            preds, _ = solve_task_v3(task)
            submission[tid] = preds
        write_json(out_path, submission)
        print(f"Wrote submission (V3) to {out_path}")
