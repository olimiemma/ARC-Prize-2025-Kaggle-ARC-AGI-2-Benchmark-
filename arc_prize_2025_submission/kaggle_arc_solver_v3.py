"""
Kaggle ARC Prize 2025 — V3 Solver (single-file)

- Self-contained, no external deps.
- Finds arc-agi_test_challenges.json under /kaggle/input.
- Writes /kaggle/working/submission.json
"""

import os, json
from collections import deque
from typing import Any, Dict, List, Tuple, Optional

Grid = List[List[int]]
Op = Tuple[str, tuple]
PaletteMap = Dict[int, int]

# ---------------
# IO
# ---------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f)

# ---------------
# Grid helpers
# ---------------
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

def block_reduce(g: Grid, hr: int, wr: int) -> Grid:
    h, w = dims(g)
    if hr <= 0 or wr <= 0: return g
    if h % hr != 0 or w % wr != 0: return g
    th, tw = h // hr, w // wr
    out: Grid = [[0 for _ in range(tw)] for _ in range(th)]
    for br in range(th):
        for bc in range(tw):
            freq: Dict[int, int] = {}
            for r in range(br * hr, (br + 1) * hr):
                for c in range(bc * wr, (bc + 1) * wr):
                    v = g[r][c]; freq[v] = freq.get(v, 0) + 1
            best, best_c = 0, 0
            for color, cnt in freq.items():
                if cnt > best:
                    best, best_c = cnt, color
            out[br][bc] = best_c
    return out

def infer_block_reduce_ratios(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Tuple[int, int]]:
    bhr: Optional[int] = None
    bwr: Optional[int] = None
    for src, dst in train_pairs:
        h1, w1 = dims(src); h2, w2 = dims(dst)
        if h2 == 0 or w2 == 0: return None
        if h1 % h2 != 0 or w1 % w2 != 0: return None
        r_h, r_w = h1 // h2, w1 // w2
        if bhr is None:
            bhr, bwr = r_h, r_w
        else:
            if bhr != r_h or bwr != r_w:
                return None
    return (bhr or 1, bwr or 1)

def remove_border_k(g: Grid, t: int) -> Grid:
    h, w = dims(g)
    if t <= 0 or h <= 2 * t or w <= 2 * t: return g
    out: Grid = []
    for r in range(t, h - t):
        out.append(g[r][t:w - t])
    return out

def add_border_k(g: Grid, t: int, color: int) -> Grid:
    if t <= 0: return g
    h, w = dims(g)
    th, tw = h + 2 * t, w + 2 * t
    out: Grid = [[color for _ in range(tw)] for _ in range(th)]
    for r in range(h):
        for c in range(w):
            out[r + t][c + t] = g[r][c]
    return out

def add_border(g: Grid, color: int) -> Grid:
    h, w = dims(g)
    out: Grid = [[color for _ in range(w + 2)] for _ in range(h + 2)]
    for r in range(h):
        for c in range(w):
            out[r + 1][c + 1] = g[r][c]
    return out

def remove_border(g: Grid) -> Optional[Grid]:
    h, w = dims(g)
    if h <= 2 or w <= 2:
        return None
    border_vals = []
    for c in range(w):
        border_vals.append(g[0][c])
        border_vals.append(g[h - 1][c])
    for r in range(h):
        border_vals.append(g[r][0])
        border_vals.append(g[r][w - 1])
    if len(set(border_vals)) != 1:
        return None
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

# ---------------
# Features
# ---------------
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
            hr, wr = r_h, r_w
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

# ---------------
# Components
# ---------------
def cc_label_color(g: Grid, target_color: int) -> List[List[int]]:
    h, w = dims(g)
    labels = [[-1 for _ in range(w)] for _ in range(h)]
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if g[r][c] == target_color and labels[r][c] == -1:
                q = deque()
                q.append((r, c))
                labels[r][c] = comp_id
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and labels[nr][nc] == -1 and g[nr][nc] == target_color:
                            labels[nr][nc] = comp_id
                            q.append((nr, nc))
                comp_id += 1
    return labels

def largest_nonbg_component_bbox(g: Grid) -> Optional[Tuple[int,int,int,int]]:
    h, w = dims(g)
    bg = most_common_color(g)
    best_size = 0
    best_bbox: Optional[Tuple[int,int,int,int]] = None
    colors: Dict[int, bool] = {}
    for r in range(h):
        for c in range(w):
            colors[g[r][c]] = True
    for color in colors.keys():
        if color == bg:
            continue
        labels = cc_label_color(g, color)
        sizes: Dict[int, int] = {}
        bboxes: Dict[int, Tuple[int,int,int,int]] = {}
        for r in range(h):
            for c in range(w):
                cid = labels[r][c]
                if cid >= 0:
                    sizes[cid] = sizes.get(cid, 0) + 1
                    if cid not in bboxes:
                        bboxes[cid] = (r, c, r, c)
                    else:
                        r0, c0, r1, c1 = bboxes[cid]
                        bboxes[cid] = (min(r0,r), min(c0,c), max(r1,r), max(c1,c))
        for cid, sz in sizes.items():
            if sz > best_size:
                best_size = sz
                best_bbox = bboxes[cid]
    return best_bbox

def crop_bbox(g: Grid, bbox: Tuple[int,int,int,int]) -> Grid:
    r0, c0, r1, c1 = bbox
    out: Grid = []
    for r in range(r0, r1+1):
        out.append(g[r][c0:c1+1])
    return out

def keep_largest_nonbg(g: Grid, fill: int) -> Grid:
    h, w = dims(g)
    bg = most_common_color(g)
    bbox = largest_nonbg_component_bbox(g)
    if bbox is None:
        return [[fill for _ in range(w)] for _ in range(h)]
    r0, c0, r1, c1 = bbox
    out = [[fill for _ in range(w)] for _ in range(h)]
    for r in range(r0, r1+1):
        for c in range(c0, c1+1):
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
    for r in range(min(ch, h)):
        for c in range(min(cw, w)):
            out[r][c] = cropped[r][c]
    return out

def extract_largest_nonbg(g: Grid) -> Optional[Tuple[Grid, Tuple[int,int]]]:
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

# ---------------
# Palette mapping
# ---------------
def infer_bijective_palette_map(src: Grid, dst: Grid) -> Optional[PaletteMap]:
    if dims(src) != dims(dst):
        return None
    mapping: Dict[int,int] = {}
    inverse: Dict[int,int] = {}
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
    inv: Dict[int,int] = {v:k for k,v in merged.items()}
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
    if dims(src) != dims(dst):
        return None
    mapping: Dict[int,int] = {}
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

# ---------------
# Program execution
# ---------------
BASE_GEOMS = ["rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"]

def apply_ops(g: Grid, ops: List[Op]) -> Grid:
    out = g
    for kind, params in ops:
        if kind == "geom":
            op = params[0]
            if op == "rot90": out = rotate90(out)
            elif op == "rot180": out = rotate180(out)
            elif op == "rot270": out = rotate270(out)
            elif op == "flip_h": out = flip_h(out)
            elif op == "flip_v": out = flip_v(out)
            elif op == "transpose": out = transpose(out)
        elif kind == "tile":
            hr, wr = params
            out = tile_scale(out, hr, wr)
        elif kind == "pad_crop":
            dh, dw, fill = params
            out = pad_crop_center(out, dh, dw, fill)
        elif kind == "block_reduce":
            hr, wr = params
            out = block_reduce(out, hr, wr)
        elif kind == "remove_border_k":
            t = params[0]
            out = remove_border_k(out, t)
        elif kind == "add_border_k":
            t, color = params
            out = add_border_k(out, t, color)
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
        elif kind == "move_largest_nonbg_by":
            dy, dx, fill = params
            out = move_largest_nonbg_by(out, dy, dx, fill)
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
        else:
            pass
    return out

def apply_program(g: Grid, ops: List[Op], pmap: Optional[PaletteMap]) -> Grid:
    out = apply_ops(g, ops)
    if pmap is not None:
        out = apply_palette(out, pmap)
    return out

# ---------------
# Fitting
# ---------------
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
        # Fallback to surjective
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

# ---------------
# Candidates and search
# ---------------
def enumerate_candidates(train_pairs: List[Tuple[Grid, Grid]]) -> List[List[Op]]:
    cands: List[List[Op]] = []
    for g in BASE_GEOMS:
        cands.append([("geom", (g,))])
    for g1 in BASE_GEOMS:
        for g2 in BASE_GEOMS:
            cands.append([("geom", (g1,)), ("geom", (g2,))])
    ratios = infer_size_ratios(train_pairs)
    deltas = infer_size_deltas(train_pairs)
    fill_votes: Dict[int,int] = {}
    for _, dst in train_pairs:
        c = most_common_color(dst)
        fill_votes[c] = fill_votes.get(c, 0) + 1
    fill = (max(fill_votes.items(), key=lambda kv: kv[1])[0]) if fill_votes else 0
    if ratios is not None:
        hr, wr = ratios
        if not (hr == 1 and wr == 1):
            cands.append([("tile",(hr,wr))])
            for g in BASE_GEOMS:
                cands.append([("tile",(hr,wr)),("geom",(g,))])
                cands.append([("geom",(g,)),("tile",(hr,wr))])
    # Downscale by block majority if input larger than output with integer ratios
    br = infer_block_reduce_ratios(train_pairs)
    if br is not None:
        bhr, bwr = br
        cands.append([("block_reduce", (bhr, bwr))])
        for g in BASE_GEOMS:
            cands.append([("block_reduce", (bhr, bwr)), ("geom", (g,))])
            cands.append([("geom", (g,)), ("block_reduce", (bhr, bwr))])
    if deltas is not None:
        dh, dw = deltas
        if not (dh == 0 and dw == 0):
            cands.append([("pad_crop",(dh,dw,fill))])
            for g in BASE_GEOMS:
                cands.append([("pad_crop",(dh,dw,fill)),("geom",(g,))])
                cands.append([("geom",(g,)),("pad_crop",(dh,dw,fill))])
    same_dims = all(dims(src) == dims(dst) for src, dst in train_pairs)
    if same_dims:
        possible: Optional[Tuple[int,int]] = None
        for dy in range(-3,4):
            for dx in range(-3,4):
                ok = True
                for src, dst in train_pairs:
                    trial = translate(src, dy, dx, most_common_color(dst))
                    if trial != dst:
                        ok = False
                        break
                if ok:
                    possible = (dy, dx); break
            if possible is not None:
                break
        if possible is not None:
            dy, dx = possible
            cands.append([("translate",(dy,dx,fill))])
            for g in BASE_GEOMS:
                cands.append([("translate",(dy,dx,fill)),("geom",(g,))])
                cands.append([("geom",(g,)),("translate",(dy,dx,fill))])
    suggest_crop = any(dims(dst)[0] <= dims(src)[0] and dims(dst)[1] <= dims(src)[1] for src, dst in train_pairs)
    if suggest_crop:
        cands.append([("crop_largest_nonbg",tuple())])
        for g in BASE_GEOMS:
            cands.append([("crop_largest_nonbg",tuple()),("geom",(g,))])
            cands.append([("geom",(g,)),("crop_largest_nonbg",tuple())])
        cands.append([("keep_largest_nonbg",(fill,))])
        for g in BASE_GEOMS:
            cands.append([("keep_largest_nonbg",(fill,)),("geom",(g,))])
            cands.append([("geom",(g,)),("keep_largest_nonbg",(fill,))])
    if deltas is not None and deltas == (-2,-2):
        cands.append([("remove_border",tuple())])
        for g in BASE_GEOMS:
            cands.append([("remove_border",tuple()),("geom",(g,))])
            cands.append([("geom",(g,)),("remove_border",tuple())])
    if deltas is not None and deltas == (2,2):
        cands.append([("add_border",(fill,))])
        for g in BASE_GEOMS:
            cands.append([("add_border",(fill,)),("geom",(g,))])
            cands.append([("geom",(g,)),("add_border",(fill,))])
    # Multi-width borders if consistent deltas suggest larger t
    if deltas is not None and deltas[0] == deltas[1] and deltas[0] % 2 == 0:
        t = abs(deltas[0]) // 2
        if deltas[0] < 0 and t > 1:
            cands.append([("remove_border_k", (t,))])
        if deltas[0] > 0 and t > 1:
            cands.append([("add_border_k", (t, fill))])
    cands.append([("move_largest_nonbg_to_tl",(fill,))])
    cands.append([("sym_h_lr",tuple())])
    cands.append([("sym_v_tb",tuple())])
    deltas_cc: Optional[Tuple[int,int]] = None
    ok = True
    for src, dst in train_pairs:
        es = extract_largest_nonbg(src); ed = extract_largest_nonbg(dst)
        if es is None or ed is None:
            ok = False; break
        (_, (rs, cs)) = es; (_, (rd, cd)) = ed
        dyy, dxx = (rd - rs, cd - cs)
        if deltas_cc is None:
            deltas_cc = (dyy, dxx)
        else:
            if deltas_cc != (dyy, dxx):
                ok = False; break
    if ok and deltas_cc is not None:
        dy, dx = deltas_cc
        cands.append([("move_largest_nonbg_by",(dy,dx,fill))])
    seen = set(); uniq: List[List[Op]] = []
    for seq in cands:
        key = tuple(seq)
        if key not in seen:
            seen.add(key); uniq.append(seq)
    def signature(ops: List[Op]) -> Tuple:
        probe: Grid = [[0,1,2],[3,4,5]]
        only_geom = [op for op in ops if op[0] == "geom"]
        out = apply_ops(probe, only_geom)
        return tuple(tuple(row) for row in out), len([op for op in ops if op[0] != "geom"])
    seen_sig = set(); final: List[List[Op]] = []
    for seq in uniq:
        sig = signature(seq)
        if sig in seen_sig: continue
        seen_sig.add(sig)
        final.append(seq)
    return final

def build_operator_pool(train_pairs: List[Tuple[Grid, Grid]]) -> List[Op]:
    pool: List[Op] = []
    for g in BASE_GEOMS: pool.append(("geom",(g,)))
    ratios = infer_size_ratios(train_pairs)
    deltas = infer_size_deltas(train_pairs)
    fill_votes: Dict[int,int] = {}
    for _, dst in train_pairs:
        c = most_common_color(dst)
        fill_votes[c] = fill_votes.get(c, 0) + 1
    fill = (max(fill_votes.items(), key=lambda kv: kv[1])[0]) if fill_votes else 0
    if ratios is not None:
        hr, wr = ratios
        if not (hr == 1 and wr == 1):
            pool.append(("tile",(hr,wr)))
    br = infer_block_reduce_ratios(train_pairs)
    if br is not None:
        bhr, bwr = br
        pool.append(("block_reduce", (bhr, bwr)))
    if deltas is not None:
        dh, dw = deltas
        if not (dh == 0 and dw == 0):
            pool.append(("pad_crop",(dh,dw,fill)))
        if deltas == (-2,-2): pool.append(("remove_border",tuple()))
        if deltas == (2,2): pool.append(("add_border",(fill,)))
        if dh == dw and dh % 2 == 0 and abs(dh) >= 4:
            t = abs(dh) // 2
            if dh < 0:
                pool.append(("remove_border_k", (t,)))
            else:
                pool.append(("add_border_k", (t, fill)))
    same_dims = all(dims(src) == dims(dst) for src, dst in train_pairs)
    if same_dims:
        found: Optional[Tuple[int,int]] = None
        for dy in range(-2,3):
            for dx in range(-2,3):
                ok = True
                for src, dst in train_pairs:
                    trial = translate(src, dy, dx, most_common_color(dst))
                    if trial != dst:
                        ok = False; break
                if ok: found = (dy,dx); break
            if found is not None: break
        if found is not None:
            dy, dx = found
            pool.append(("translate",(dy,dx,fill)))
    pool += [("crop_largest_nonbg",tuple()), ("keep_largest_nonbg",(fill,)), ("move_largest_nonbg_to_tl",(fill,)), ("sym_h_lr",tuple()), ("sym_v_tb",tuple())]
    seen = set(); uniq: List[Op] = []
    for op in pool:
        if op not in seen:
            seen.add(op); uniq.append(op)
    return uniq

def beam_search_fit(train_pairs: List[Tuple[Grid, Grid]], pool: List[Op], beam: int = 12, depth: int = 3, allow_palette: bool = True) -> Optional[Tuple[List[Op], Optional[PaletteMap]]]:
    frontier: List[List[Op]] = [[]]
    tried = set()
    for d in range(1, depth+1):
        new_frontier: List[List[Op]] = []
        for seq in frontier:
            for op in pool:
                new_seq = seq + [op]
                key = tuple(new_seq)
                if key in tried: continue
                tried.add(key)
                fit = fit_program(train_pairs, new_seq, allow_palette=allow_palette)
                if fit is not None:
                    return fit
                new_frontier.append(new_seq)
                if len(new_frontier) > beam:
                    new_frontier = new_frontier[:beam]
        frontier = new_frontier
        if not frontier: break
    return None

# ---------------
# Solve task
# ---------------
def solve_task(task: Dict[str, Any]) -> Tuple[List[Dict[str, Grid]], Dict[str, Any]]:
    train = task.get("train", [])
    test = task.get("test", [])
    train_pairs: List[Tuple[Grid, Grid]] = [(t["input"], t["output"]) for t in train]
    candidates = enumerate_candidates(train_pairs)

    attempt1: Optional[Tuple[List[Op], Optional[PaletteMap]]] = None
    attempt2: Optional[Tuple[List[Op], Optional[PaletteMap]]] = None
    found_programs: List[Tuple[List[Op], Optional[PaletteMap]]] = []

    # Attempt 1: palette-first
    pmap = fit_palette_only(train_pairs)
    if pmap is not None:
        attempt1 = ([], pmap)
        found_programs.append(attempt1)
    if attempt1 is None:
        for seq in candidates:
            fit = fit_program(train_pairs, seq, allow_palette=True)
            if fit is not None:
                attempt1 = fit; found_programs.append(fit)
                # keep collecting a few more diverse fits
                if len(found_programs) >= 3:
                    break
    if attempt1 is None:
        pool = build_operator_pool(train_pairs)
        fit = beam_search_fit(train_pairs, pool, beam=12, depth=3, allow_palette=True)
        if fit is not None:
            attempt1 = fit; found_programs.append(fit)

    # Attempt 2: geometry/components-first
    for seq in candidates:
        fit = fit_program(train_pairs, seq, allow_palette=False)
        if fit is not None:
            # add if distinct
            if all(not (fp[0] == fit[0] and fp[1] == fit[1]) for fp in found_programs):
                found_programs.append(fit)
                if attempt2 is None:
                    attempt2 = fit
            if len(found_programs) >= 4:
                break
    if attempt2 is None:
        for seq in candidates:
            fit = fit_program(train_pairs, seq, allow_palette=True)
            if fit is not None:
                if all(not (fp[0] == fit[0] and fp[1] == fit[1]) for fp in found_programs):
                    found_programs.append(fit)
                    if attempt2 is None:
                        attempt2 = fit
                if len(found_programs) >= 5:
                    break
    if attempt2 is None:
        pool = build_operator_pool(train_pairs)
        fit = beam_search_fit(train_pairs, pool, beam=12, depth=3, allow_palette=False)
        if fit is None:
            fit = beam_search_fit(train_pairs, pool, beam=12, depth=3, allow_palette=True)
        if fit is not None and all(not (fp[0] == fit[0] and fp[1] == fit[1]) for fp in found_programs):
            found_programs.append(fit)
            if attempt2 is None:
                attempt2 = fit

    preds: List[Dict[str, Grid]] = []
    for item in test:
        tin: Grid = item.get("input", [])
        # choose two diverse programs if available
        prog1 = attempt1
        prog2 = attempt2
        if prog1 is None and found_programs:
            prog1 = found_programs[0]
        if prog2 is None and len(found_programs) > 1:
            prog2 = found_programs[1]
        a1 = apply_program(tin, prog1[0], prog1[1]) if prog1 is not None else tin
        a2 = apply_program(tin, prog2[0], prog2[1]) if prog2 is not None else (
            rotate180(tin) if dims(tin) == dims(rotate180(tin)) else tin
        )
        preds.append({"attempt_1": a1, "attempt_2": a2})

    debug = {"attempt_1": attempt1, "attempt_2": attempt2}
    return preds, debug

# ---------------
# Main: find test JSON and write submission
# ---------------
def find_test_json() -> str:
    target = "arc-agi_test_challenges.json"
    for root, _, files in os.walk("/kaggle/input"):
        if target in files:
            return os.path.join(root, target)
    default = "/kaggle/input/arc-prize-2025/arc-prize-2025/data/arc-agi_test_challenges.json"
    if os.path.exists(default):
        return default
    raise FileNotFoundError("Could not locate arc-agi_test_challenges.json under /kaggle/input.")

def main():
    test_path = find_test_json()
    print(f"Using test file: {test_path}")
    challenges = load_json(test_path)
    submission: Dict[str, Any] = {}
    for tid, task in challenges.items():
        preds, _ = solve_task(task)
        submission[tid] = preds
    out_path = "/kaggle/working/submission.json"
    write_json(out_path, submission)
    print(f"Wrote submission to {out_path}")

if __name__ == "__main__":
    main()
