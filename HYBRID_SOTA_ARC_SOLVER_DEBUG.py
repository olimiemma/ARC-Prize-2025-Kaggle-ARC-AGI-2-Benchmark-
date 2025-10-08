"""
═══════════════════════════════════════════════════════════════════════════════
 HYBRID SOTA ARC SOLVER - DEBUG VERSION
═══════════════════════════════════════════════════════════════════════════════

This version includes EXTENSIVE LOGGING to identify the exact failure point.
Use this to debug, then transfer fixes to production version.

DEBUG FEATURES:
- Exception tracing with full stack traces
- Detailed logging at every step
- DSL operation testing at startup
- Limited to 10 tasks for fast testing
- Progress indicators for every tier

═══════════════════════════════════════════════════════════════════════════════
"""

# KAGGLE SETUP - Install bitsandbytes for wb55L model
# Uncomment this line in Kaggle:
# !pip install bitsandbytes

import os
import json
import time
import numpy as np
import traceback
from typing import List, Dict, Tuple, Optional, Any
from collections import deque, Counter
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers (for LLM)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except:
    HAS_TRANSFORMERS = False
    print("⚠️ Transformers not available - LLM tier will be disabled")

print("🔍 Initializing HYBRID SOTA ARC Solver - DEBUG VERSION")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Global configuration"""
    # Time budgets per tier (seconds)
    TIER1_TIMEOUT = 5   # Fast symbolic
    TIER2_TIMEOUT = 15  # Beam search
    TIER3_TIMEOUT = 30  # LLM + TTA

    # Search parameters
    BEAM_WIDTH = 12
    BEAM_DEPTH = 3

    # LLM parameters
    LLM_MAX_ATTEMPTS = 5
    LLM_MAX_TOKENS = 512
    LLM_TEMPERATURE = 0.7

    # Model paths - FIXED
    MODEL_PATH = "/kaggle/input/wb55l_nemomini_fulleval/transformers/default/1"  # Corrected
    FALLBACK_MODEL = "/kaggle/input/qwen-3/transformers/0.6b"  # FIXED PATH!

    # Data path
    DATA_PATH = "/kaggle/input/arc-prize-2025"
    EVAL_FILE = "arc-agi_evaluation_challenges.json"

    # Progress reporting
    REPORT_EVERY = 5  # More frequent for debugging

    # DEBUG MODE
    DEBUG_MODE = True
    MAX_TEST_TASKS = 10  # Test on 10 tasks only

# ═══════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

Grid = np.ndarray
Op = Tuple[str, tuple]
PaletteMap = Dict[int, int]
Program = Tuple[List[Op], Optional[PaletteMap]]

# ═══════════════════════════════════════════════════════════════════════════
# TIER 0: CORE DSL OPERATIONS (from a_v4.py)
# ═══════════════════════════════════════════════════════════════════════════

class DSL:
    """Domain-Specific Language - All transformations from a_v4.py"""

    # --- Geometric Operations ---
    @staticmethod
    def rotate90(g: Grid) -> Grid:
        return np.rot90(g, k=-1)

    @staticmethod
    def rotate180(g: Grid) -> Grid:
        return np.rot90(g, k=2)

    @staticmethod
    def rotate270(g: Grid) -> Grid:
        return np.rot90(g, k=1)

    @staticmethod
    def flip_h(g: Grid) -> Grid:
        return np.fliplr(g)

    @staticmethod
    def flip_v(g: Grid) -> Grid:
        return np.flipud(g)

    @staticmethod
    def transpose(g: Grid) -> Grid:
        return g.T

    # --- Scaling Operations ---
    @staticmethod
    def tile_scale(g: Grid, hr: int, wr: int) -> Grid:
        if hr <= 0 or wr <= 0:
            return g
        return np.kron(g, np.ones((hr, wr), dtype=g.dtype))

    @staticmethod
    def block_reduce(g: Grid, hr: int, wr: int) -> Grid:
        h, w = g.shape
        if hr <= 0 or wr <= 0 or h % hr != 0 or w % wr != 0:
            return g
        th, tw = h // hr, w // wr
        out = np.zeros((th, tw), dtype=g.dtype)
        for r in range(th):
            for c in range(tw):
                block = g[r*hr:(r+1)*hr, c*wr:(c+1)*wr]
                out[r, c] = np.bincount(block.flatten()).argmax()
        return out

    # --- Padding/Cropping ---
    @staticmethod
    def pad_crop_center(g: Grid, dh: int, dw: int, fill: int) -> Grid:
        h, w = g.shape
        th, tw = max(1, h + dh), max(1, w + dw)
        out = np.full((th, tw), fill, dtype=g.dtype)

        src_top = max(0, (h - th) // 2)
        src_left = max(0, (w - tw) // 2)
        dst_top = max(0, (th - h) // 2)
        dst_left = max(0, (tw - w) // 2)

        copy_h = min(h, th)
        copy_w = min(w, tw)

        out[dst_top:dst_top+copy_h, dst_left:dst_left+copy_w] = \
            g[src_top:src_top+copy_h, src_left:src_left+copy_w]
        return out

    @staticmethod
    def add_border_k(g: Grid, t: int, color: int) -> Grid:
        if t <= 0:
            return g
        return np.pad(g, pad_width=t, mode='constant', constant_values=color)

    @staticmethod
    def remove_border_k(g: Grid, t: int) -> Grid:
        h, w = g.shape
        if t <= 0 or h <= 2 * t or w <= 2 * t:
            return g
        return g[t:-t, t:-t]

    # --- Translation ---
    @staticmethod
    def translate(g: Grid, dy: int, dx: int, fill: int) -> Grid:
        out = np.full_like(g, fill)
        h, w = g.shape
        src_r_start = max(0, -dy)
        src_r_end = min(h, h - dy)
        src_c_start = max(0, -dx)
        src_c_end = min(w, w - dx)
        dst_r_start = max(0, dy)
        dst_r_end = min(h, h + dy)
        dst_c_start = max(0, dx)
        dst_c_end = min(w, w + dx)

        out[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
            g[src_r_start:src_r_end, src_c_start:src_c_end]
        return out

    # --- Symmetry ---
    @staticmethod
    def symmetrize_h_left_to_right(g: Grid) -> Grid:
        h, w = g.shape
        hw = w // 2
        out = g.copy()
        out[:, hw:] = np.fliplr(out[:, :hw])
        return out

    @staticmethod
    def symmetrize_v_top_to_bottom(g: Grid) -> Grid:
        h, w = g.shape
        hh = h // 2
        out = g.copy()
        out[hh:, :] = np.flipud(out[:hh, :])
        return out

    # --- Palette Operations ---
    @staticmethod
    def apply_palette(g: Grid, mapping: PaletteMap) -> Grid:
        out = g.copy()
        for old_c, new_c in mapping.items():
            out[g == old_c] = new_c
        return out

    # --- Object Operations ---
    @staticmethod
    def most_common_color(g: Grid) -> int:
        if g.size == 0:
            return 0
        return int(np.bincount(g.flatten()).argmax())

    @staticmethod
    def get_largest_nonbg_bbox(g: Grid) -> Optional[Tuple[int, int, int, int]]:
        """Find bounding box of largest non-background object"""
        bg = DSL.most_common_color(g)
        h, w = g.shape
        visited = np.zeros((h, w), dtype=bool)

        best_size = 0
        best_bbox = None

        def bfs(sr: int, sc: int) -> Tuple[int, int, int, int, int]:
            queue = deque([(sr, sc)])
            visited[sr, sc] = True
            r0, c0, r1, c1 = sr, sc, sr, sc
            size = 0
            while queue:
                r, c = queue.popleft()
                size += 1
                r0, c0 = min(r0, r), min(c0, c)
                r1, c1 = max(r1, r), max(c1, c)
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] != bg:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            return (r0, c0, r1, c1, size)

        for r in range(h):
            for c in range(w):
                if not visited[r, c] and g[r, c] != bg:
                    r0, c0, r1, c1, size = bfs(r, c)
                    if size > best_size:
                        best_size = size
                        best_bbox = (r0, c0, r1, c1)

        return best_bbox

    @staticmethod
    def crop_to_largest_nonbg(g: Grid) -> Grid:
        bbox = DSL.get_largest_nonbg_bbox(g)
        if bbox is None:
            return g
        r0, c0, r1, c1 = bbox
        return g[r0:r1+1, c0:c1+1]

    @staticmethod
    def keep_largest_nonbg(g: Grid, fill: int) -> Grid:
        bbox = DSL.get_largest_nonbg_bbox(g)
        if bbox is None:
            return g
        r0, c0, r1, c1 = bbox
        bg = DSL.most_common_color(g)  # FIX: Calculate background color
        out = np.full_like(g, fill)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if g[r, c] != bg:
                    out[r, c] = g[r, c]
        return out

# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: SYMBOLIC SOLVER (from a_v4.py)
# ═══════════════════════════════════════════════════════════════════════════

class SymbolicSolver:
    """Fast symbolic solver using feature-driven search"""

    BASE_GEOMS = ["rotate90", "rotate180", "rotate270", "flip_h", "flip_v", "transpose"]

    def __init__(self):
        self.dsl = DSL()

    def solve(self, train_pairs: List[Tuple[Grid, Grid]],
              test_input: Grid, timeout: float = 5.0) -> Optional[Program]:
        """
        Tier 1: Fast symbolic solving
        Returns verified program or None
        """
        if Config.DEBUG_MODE:
            print(f"    [TIER 1] Starting solve (timeout={timeout}s)")
            print(f"    [TIER 1] Train pairs: {len(train_pairs)}")

        start_time = time.time()

        # Strategy 1: Palette-only (fastest)
        if Config.DEBUG_MODE:
            print(f"    [TIER 1] Trying palette-only...")

        result = self._try_palette_only(train_pairs)
        if result and time.time() - start_time < timeout:
            if self._verify(result, train_pairs):
                if Config.DEBUG_MODE:
                    print(f"    [TIER 1] ✓ Palette-only SUCCESS!")
                return result
            elif Config.DEBUG_MODE:
                print(f"    [TIER 1] Palette generated but failed verification")

        # Strategy 2: Feature-driven candidates
        if Config.DEBUG_MODE:
            print(f"    [TIER 1] Generating feature-driven candidates...")

        candidates = self._enumerate_candidates(train_pairs)

        if Config.DEBUG_MODE:
            print(f"    [TIER 1] Generated {len(candidates)} candidates")

        for idx, seq in enumerate(candidates):
            if time.time() - start_time >= timeout:
                if Config.DEBUG_MODE:
                    print(f"    [TIER 1] Timeout after {idx} candidates")
                break

            # Try without palette
            fit = self._fit_program(train_pairs, seq, allow_palette=False)
            if fit:
                if Config.DEBUG_MODE:
                    print(f"    [TIER 1] ✓ SUCCESS without palette (candidate {idx})")
                return fit

            # Try with palette
            fit = self._fit_program(train_pairs, seq, allow_palette=True)
            if fit:
                if Config.DEBUG_MODE:
                    print(f"    [TIER 1] ✓ SUCCESS with palette (candidate {idx})")
                return fit

        if Config.DEBUG_MODE:
            print(f"    [TIER 1] ✗ All candidates failed")

        return None

    def _try_palette_only(self, pairs: List[Tuple[Grid, Grid]]) -> Optional[Program]:
        """Try pure palette mapping (O(1) check)"""
        global_map: PaletteMap = {}
        for src, dst in pairs:
            local = self._infer_palette(src, dst)
            if local is None:
                return None
            merged = self._merge_palettes(global_map, local)
            if merged is None:
                return None
            global_map = merged
        return ([], global_map) if global_map else None

    def _enumerate_candidates(self, pairs: List[Tuple[Grid, Grid]]) -> List[List[Op]]:
        """Generate feature-driven candidates (from a_v4.py logic)"""
        cands: List[List[Op]] = []

        # Single geometry operations
        for g in self.BASE_GEOMS:
            cands.append([("geom", (g,))])

        # Double geometry operations
        for g1 in self.BASE_GEOMS:
            for g2 in self.BASE_GEOMS[:3]:  # Limit combinations
                cands.append([("geom", (g1,)), ("geom", (g2,))])

        # Feature-based operations
        ratios = self._infer_size_ratios(pairs)
        deltas = self._infer_size_deltas(pairs)
        fill = self.dsl.most_common_color(pairs[0][1]) if pairs else 0

        if ratios and ratios != (1, 1):
            cands.append([("tile", ratios)])
            for g in self.BASE_GEOMS[:3]:
                cands.append([("tile", ratios), ("geom", (g,))])

        if deltas and deltas != (0, 0):
            cands.append([("pad_crop", (*deltas, fill))])
            for g in self.BASE_GEOMS[:3]:
                cands.append([("pad_crop", (*deltas, fill)), ("geom", (g,))])

        # Object operations
        cands.append([("crop_largest_nonbg", ())])
        cands.append([("keep_largest_nonbg", (fill,))])

        # Symmetry
        cands.append([("sym_h_lr", ())])
        cands.append([("sym_v_tb", ())])

        return cands[:50]  # Limit for speed

    def _fit_program(self, pairs: List[Tuple[Grid, Grid]],
                     ops: List[Op], allow_palette: bool) -> Optional[Program]:
        """Check if program works on all training pairs"""
        try:
            temp_grids = [self._apply_ops(src, ops) for src, _ in pairs]
        except Exception as e:
            if Config.DEBUG_MODE:
                print(f"      [ERROR] Exception in _apply_ops: {e}")
            return None

        if not allow_palette:
            if all(np.array_equal(g, dst) for g, (_, dst) in zip(temp_grids, pairs)):
                return (ops, None)
            return None

        # Try to fit palette
        global_map: PaletteMap = {}
        for temp_g, (_, dst) in zip(temp_grids, pairs):
            local_map = self._infer_palette(temp_g, dst)
            if local_map is None:
                return None
            merged = self._merge_palettes(global_map, local_map)
            if merged is None:
                return None
            global_map = merged

        return (ops, global_map)

    def _apply_ops(self, g: Grid, ops: List[Op]) -> Grid:
        """Execute operation sequence"""
        out = g
        for kind, params in ops:
            if kind == "geom":
                out = getattr(self.dsl, params[0])(out)
            elif kind == "tile":
                out = self.dsl.tile_scale(out, *params)
            elif kind == "pad_crop":
                out = self.dsl.pad_crop_center(out, *params)
            elif kind == "crop_largest_nonbg":
                out = self.dsl.crop_to_largest_nonbg(out)
            elif kind == "keep_largest_nonbg":
                out = self.dsl.keep_largest_nonbg(out, *params)
            elif kind == "sym_h_lr":
                out = self.dsl.symmetrize_h_left_to_right(out)
            elif kind == "sym_v_tb":
                out = self.dsl.symmetrize_v_top_to_bottom(out)
        return out

    def apply_program(self, g: Grid, prog: Program) -> Grid:
        """Apply complete program (ops + palette)"""
        ops, pmap = prog
        out = self._apply_ops(g, ops)
        if pmap:
            out = self.dsl.apply_palette(out, pmap)
        return out

    def _verify(self, prog: Program, pairs: List[Tuple[Grid, Grid]]) -> bool:
        """Exact verification"""
        if Config.DEBUG_MODE:
            print(f"      [VERIFY] Testing program on {len(pairs)} training pairs")

        for idx, (src, dst) in enumerate(pairs):
            try:
                pred = self.apply_program(src, prog)
                match = np.array_equal(pred, dst)

                if Config.DEBUG_MODE:
                    print(f"        Pair {idx}: {'PASS' if match else 'FAIL'} (pred {pred.shape}, dst {dst.shape})")

                if not match:
                    return False
            except Exception as e:
                if Config.DEBUG_MODE:
                    print(f"        Pair {idx}: EXCEPTION - {e}")
                return False

        return True

    def _infer_size_ratios(self, pairs: List[Tuple[Grid, Grid]]) -> Optional[Tuple[int, int]]:
        ratios = [(d.shape[0] / s.shape[0], d.shape[1] / s.shape[1])
                  for s, d in pairs if s.size > 0]
        if not ratios or not all(r[0] == ratios[0][0] and r[1] == ratios[0][1] for r in ratios):
            return None
        hr, wr = ratios[0]
        return (int(hr), int(wr)) if hr.is_integer() and wr.is_integer() else None

    def _infer_size_deltas(self, pairs: List[Tuple[Grid, Grid]]) -> Optional[Tuple[int, int]]:
        deltas = [(d.shape[0] - s.shape[0], d.shape[1] - s.shape[1]) for s, d in pairs]
        return deltas[0] if all(d == deltas[0] for d in deltas) else None

    def _infer_palette(self, src: Grid, dst: Grid) -> Optional[PaletteMap]:
        if src.shape != dst.shape:
            return None
        mapping: Dict[int, int] = {}
        inverse: Dict[int, int] = {}
        for s, d in zip(src.flatten(), dst.flatten()):
            if s in mapping and mapping[s] != d:
                return None
            if d in inverse and inverse[d] != s:
                return None
            mapping[s] = d
            inverse[d] = s
        return mapping

    def _merge_palettes(self, g_map: PaletteMap, l_map: PaletteMap) -> Optional[PaletteMap]:
        merged = g_map.copy()
        inv = {v: k for k, v in merged.items()}
        for k, v in l_map.items():
            if k in merged and merged[k] != v:
                return None
            if v in inv and inv[v] != k:
                return None
            merged[k] = v
            inv[v] = k
        return merged

# ═══════════════════════════════════════════════════════════════════════════
# TIER 2: BEAM SEARCH SOLVER (from a_v4.py)
# ═══════════════════════════════════════════════════════════════════════════

class BeamSearchSolver:
    """Compositional beam search for harder tasks"""

    def __init__(self, beam_width=12, depth=3):
        self.symbolic = SymbolicSolver()
        self.beam_width = beam_width
        self.depth = depth

    def solve(self, train_pairs: List[Tuple[Grid, Grid]],
              test_input: Grid, timeout: float = 15.0) -> Optional[Program]:
        """
        Tier 2: Beam search
        More expensive but finds compositional solutions
        """
        if Config.DEBUG_MODE:
            print(f"    [TIER 2] Starting beam search (width={self.beam_width}, depth={self.depth})")

        start_time = time.time()

        pool = self._build_operator_pool(train_pairs)

        if Config.DEBUG_MODE:
            print(f"    [TIER 2] Operator pool size: {len(pool)}")

        frontier: List[List[Op]] = [[]]
        tried = {tuple()}

        for d in range(1, self.depth + 1):
            if time.time() - start_time >= timeout:
                if Config.DEBUG_MODE:
                    print(f"    [TIER 2] Timeout at depth {d}")
                break

            if Config.DEBUG_MODE:
                print(f"    [TIER 2] Exploring depth {d}, frontier size: {len(frontier)}")

            new_frontier: List[List[Op]] = []
            for seq in frontier:
                for op in pool:
                    if time.time() - start_time >= timeout:
                        break

                    new_seq = seq + [op]
                    key = tuple(new_seq)
                    if key in tried:
                        continue
                    tried.add(key)

                    # Try with palette
                    fit = self.symbolic._fit_program(train_pairs, new_seq, allow_palette=True)
                    if fit:
                        if Config.DEBUG_MODE:
                            print(f"    [TIER 2] ✓ SUCCESS at depth {d}")
                        return fit

                    new_frontier.append(new_seq)

            # Prune to beam width
            frontier = sorted(new_frontier, key=len)[:self.beam_width]
            if not frontier:
                if Config.DEBUG_MODE:
                    print(f"    [TIER 2] Frontier empty at depth {d}")
                break

        if Config.DEBUG_MODE:
            print(f"    [TIER 2] ✗ Search exhausted")

        return None

    def _build_operator_pool(self, pairs: List[Tuple[Grid, Grid]]) -> List[Op]:
        """Build feature-driven operator pool"""
        pool: List[Op] = []

        # Geometry
        for g in self.symbolic.BASE_GEOMS:
            pool.append(("geom", (g,)))

        # Feature-based
        ratios = self.symbolic._infer_size_ratios(pairs)
        deltas = self.symbolic._infer_size_deltas(pairs)
        fill = DSL.most_common_color(pairs[0][1]) if pairs else 0

        if ratios and ratios != (1, 1):
            pool.append(("tile", ratios))
        if deltas and deltas != (0, 0):
            pool.append(("pad_crop", (*deltas, fill)))

        # Objects
        pool.append(("crop_largest_nonbg", ()))
        pool.append(("keep_largest_nonbg", (fill,)))

        # Symmetry
        pool.append(("sym_h_lr", ()))
        pool.append(("sym_v_tb", ()))

        return pool

# ═══════════════════════════════════════════════════════════════════════════
# TIER 3: LLM + TTA SOLVER (DISABLED FOR DEBUG)
# ═══════════════════════════════════════════════════════════════════════════

class LLMTTASolver:
    """LLM-based solver - DISABLED IN DEBUG MODE"""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.has_llm = (model is not None and tokenizer is not None)
        self.symbolic = SymbolicSolver()

    def solve(self, train_pairs, test_input, timeout=30.0, max_attempts=5):
        print("    [TIER 3] DISABLED IN DEBUG MODE")
        return None

# ═══════════════════════════════════════════════════════════════════════════
# HYBRID ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class HybridSolver:
    """Multi-tier solver combining symbolic and LLM approaches"""

    def __init__(self, llm_model=None, llm_tokenizer=None):
        self.tier1 = SymbolicSolver()
        self.tier2 = BeamSearchSolver(
            beam_width=Config.BEAM_WIDTH,
            depth=Config.BEAM_DEPTH
        )
        self.tier3 = LLMTTASolver(llm_model, llm_tokenizer)

        self.stats = {
            'tier1_success': 0,
            'tier2_success': 0,
            'tier3_success': 0,
            'failures': 0
        }

    def solve_task(self, task: Dict[str, Any]) -> List[Dict[str, List[List[int]]]]:
        """
        Solve a single ARC task using multi-tier strategy
        Returns pass@2 predictions
        """
        if Config.DEBUG_MODE:
            print(f"  [TASK] Converting training pairs...")

        # FIX: Ensure consistent dtype
        train_pairs = [(np.array(ex['input'], dtype=int),
                       np.array(ex['output'], dtype=int))
                      for ex in task['train']]

        if Config.DEBUG_MODE:
            print(f"  [TASK] Train pairs: {len(train_pairs)}")
            for idx, (inp, out) in enumerate(train_pairs):
                print(f"    Pair {idx}: input {inp.shape} dtype={inp.dtype}, output {out.shape} dtype={out.dtype}")

        predictions = []

        for test_idx, test_case in enumerate(task['test']):
            if Config.DEBUG_MODE:
                print(f"  [TASK] Solving test case {test_idx + 1}/{len(task['test'])}")

            test_input = np.array(test_case['input'], dtype=int)

            if Config.DEBUG_MODE:
                print(f"    Test input: {test_input.shape} dtype={test_input.dtype}")

            attempt1, attempt2 = self._solve_with_tiers(train_pairs, test_input)
            predictions.append({
                'attempt_1': attempt1.tolist(),
                'attempt_2': attempt2.tolist()
            })

        return predictions

    def _solve_with_tiers(self, train_pairs: List[Tuple[Grid, Grid]],
                          test_input: Grid) -> Tuple[Grid, Grid]:
        """
        Progressive tier execution
        Returns (attempt_1, attempt_2) grids
        """
        programs = []

        # Tier 1: Fast symbolic (5 sec)
        if Config.DEBUG_MODE:
            print(f"  [ORCHESTRATOR] Trying Tier 1...")

        try:
            prog = self.tier1.solve(train_pairs, test_input, timeout=Config.TIER1_TIMEOUT)
            if prog:
                programs.append(prog)
                self.stats['tier1_success'] += 1
                if Config.DEBUG_MODE:
                    print(f"  [ORCHESTRATOR] ✓ Tier 1 SUCCESS")
        except Exception as e:
            if Config.DEBUG_MODE:
                print(f"  [ORCHESTRATOR] ✗ Tier 1 EXCEPTION: {e}")
                traceback.print_exc()

        # Tier 2: Beam search (15 sec) - only if tier 1 failed
        if not programs:
            if Config.DEBUG_MODE:
                print(f"  [ORCHESTRATOR] Trying Tier 2...")

            try:
                prog = self.tier2.solve(train_pairs, test_input, timeout=Config.TIER2_TIMEOUT)
                if prog:
                    programs.append(prog)
                    self.stats['tier2_success'] += 1
                    if Config.DEBUG_MODE:
                        print(f"  [ORCHESTRATOR] ✓ Tier 2 SUCCESS")
            except Exception as e:
                if Config.DEBUG_MODE:
                    print(f"  [ORCHESTRATOR] ✗ Tier 2 EXCEPTION: {e}")
                    traceback.print_exc()

        # Tier 3: DISABLED IN DEBUG
        # (Will enable after fixing symbolic solver)

        # Generate pass@2 attempts
        if len(programs) >= 1:
            if Config.DEBUG_MODE:
                print(f"  [ORCHESTRATOR] Applying program to test input...")

            try:
                attempt1 = self.tier1.apply_program(test_input, programs[0])

                if len(programs) >= 2:
                    attempt2 = self.tier1.apply_program(test_input, programs[1])
                else:
                    # Mutate: try with a simple rotation
                    mutated_prog = (programs[0][0] + [("geom", ("rotate90",))], programs[0][1])
                    attempt2 = self.tier1.apply_program(test_input, mutated_prog)

                if Config.DEBUG_MODE:
                    print(f"  [ORCHESTRATOR] ✓ Generated pass@2 attempts")
            except Exception as e:
                if Config.DEBUG_MODE:
                    print(f"  [ORCHESTRATOR] ✗ Program application EXCEPTION: {e}")
                    traceback.print_exc()
                attempt1 = test_input
                attempt2 = test_input
                self.stats['failures'] += 1
        else:
            # Fallback: return input
            if Config.DEBUG_MODE:
                print(f"  [ORCHESTRATOR] ✗ No programs found, using fallback")

            attempt1 = test_input
            attempt2 = test_input
            self.stats['failures'] += 1

        return attempt1, attempt2

# ═══════════════════════════════════════════════════════════════════════════
# DSL TESTING
# ═══════════════════════════════════════════════════════════════════════════

def test_dsl_operations():
    """Test DSL operations at startup"""
    print("\n" + "=" * 80)
    print("  🧪 TESTING DSL OPERATIONS")
    print("=" * 80)

    test_grid = np.array([[1, 2], [3, 4]], dtype=int)
    print(f"\nTest grid:\n{test_grid}")

    try:
        print(f"\nrotate90:\n{DSL.rotate90(test_grid)}")
        print(f"\nrotate180:\n{DSL.rotate180(test_grid)}")
        print(f"\nflip_h:\n{DSL.flip_h(test_grid)}")
        print(f"\nflip_v:\n{DSL.flip_v(test_grid)}")
        print(f"\ntranspose:\n{DSL.transpose(test_grid)}")

        print(f"\ntile_scale(2,2):\n{DSL.tile_scale(test_grid, 2, 2)}")

        print(f"\nmost_common_color: {DSL.most_common_color(test_grid)}")

        print("\n✅ All basic DSL operations working!")
    except Exception as e:
        print(f"\n❌ DSL TEST FAILED: {e}")
        traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING (DISABLED FOR DEBUG - Focus on Symbolic First)
# ═══════════════════════════════════════════════════════════════════════════

def load_llm_model():
    """LLM loading DISABLED in debug mode"""
    print("\n⚠️ LLM loading DISABLED in debug mode (focusing on symbolic solver)")
    return None, None

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def find_evaluation_file():
    """Locate evaluation challenges file"""
    paths_to_try = [
        os.path.join(Config.DATA_PATH, Config.EVAL_FILE),
        os.path.join(Config.DATA_PATH, "arc-prize-2025", Config.EVAL_FILE),
        os.path.join(Config.DATA_PATH, "data", Config.EVAL_FILE),
        f"/kaggle/input/arc-prize-2025/{Config.EVAL_FILE}",
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Could not locate {Config.EVAL_FILE}")

def main():
    """Main execution loop - DEBUG VERSION"""
    print("\n" + "=" * 80)
    print("  🎯 HYBRID SOTA ARC SOLVER - DEBUG MODE")
    print("=" * 80 + "\n")

    # Test DSL operations first
    test_dsl_operations()

    # Skip LLM loading in debug mode
    model, tokenizer = load_llm_model()

    # Initialize solver
    print("\n🔧 Initializing Hybrid Solver...")
    solver = HybridSolver(llm_model=model, llm_tokenizer=tokenizer)
    print("✅ Solver ready!")
    print(f"  - Tier 1: Fast Symbolic ({Config.TIER1_TIMEOUT}s timeout)")
    print(f"  - Tier 2: Beam Search ({Config.TIER2_TIMEOUT}s timeout, beam={Config.BEAM_WIDTH}, depth={Config.BEAM_DEPTH})")
    print(f"  - Tier 3: DISABLED (debug mode)")

    # Load evaluation tasks
    print(f"\n📂 Loading evaluation tasks...")
    eval_path = find_evaluation_file()
    print(f"  Using: {eval_path}")

    with open(eval_path, 'r') as f:
        tasks = json.load(f)

    total_tasks = len(tasks)
    print(f"✅ Loaded {total_tasks} tasks")
    print(f"⚠️  DEBUG MODE: Testing first {Config.MAX_TEST_TASKS} tasks only\n")

    # Solve all tasks
    print("=" * 80)
    print("  🧠 Solving Tasks (DEBUG MODE)")
    print("=" * 80 + "\n")

    submission = {}
    start_time = time.time()

    for task_idx, (task_id, task_data) in enumerate(tasks.items(), 1):
        # DEBUG: Limit to first N tasks
        if task_idx > Config.MAX_TEST_TASKS:
            print(f"\n⚠️  Stopping at {Config.MAX_TEST_TASKS} tasks (DEBUG MODE)")
            break

        task_start = time.time()

        print(f"\n{'='*80}")
        print(f"TASK {task_idx}/{min(total_tasks, Config.MAX_TEST_TASKS)}: {task_id}")
        print(f"{'='*80}")

        try:
            predictions = solver.solve_task(task_data)
            submission[task_id] = predictions

            # Task completion
            task_time = time.time() - task_start
            print(f"\n✓ Task {task_idx} complete: {task_id[:16]}... ({task_time:.1f}s)")

        except Exception as e:
            print(f"\n❌ TASK {task_idx} CRITICAL ERROR:")
            print(f"Exception: {e}")
            traceback.print_exc()

            # Fallback: return input for all test cases
            fallback = [{'attempt_1': test['input'], 'attempt_2': test['input']}
                       for test in task_data.get('test', [])]
            submission[task_id] = fallback
            solver.stats['failures'] += 1

        # Progress reporting
        if task_idx % Config.REPORT_EVERY == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / task_idx
            remaining = (min(total_tasks, Config.MAX_TEST_TASKS) - task_idx) * avg_time
            print(f"\n📊 Progress: {task_idx}/{min(total_tasks, Config.MAX_TEST_TASKS)} tasks")
            print(f"  ⏱️  Elapsed: {elapsed/60:.1f}min, Est. remaining: {remaining/60:.1f}min")
            print(f"  📈 Stats: T1={solver.stats['tier1_success']}, "
                  f"T2={solver.stats['tier2_success']}, "
                  f"T3={solver.stats['tier3_success']}, "
                  f"Fail={solver.stats['failures']}")

    # Final stats
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("  ✅ DEBUG RUN COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Final Statistics:")
    print(f"  Tasks tested: {min(total_tasks, Config.MAX_TEST_TASKS)}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Avg time/task: {total_time/min(total_tasks, Config.MAX_TEST_TASKS):.1f}s")
    print(f"\n  Tier 1 (Fast Symbolic): {solver.stats['tier1_success']} tasks")
    print(f"  Tier 2 (Beam Search): {solver.stats['tier2_success']} tasks")
    print(f"  Tier 3 (LLM + TTA): {solver.stats['tier3_success']} tasks (disabled)")
    print(f"  Failures (fallback): {solver.stats['failures']} tasks")

    # Success rate
    solved = solver.stats['tier1_success'] + solver.stats['tier2_success'] + solver.stats['tier3_success']
    success_rate = (solved / min(total_tasks, Config.MAX_TEST_TASKS)) * 100 if total_tasks > 0 else 0
    print(f"\n  Success rate: {success_rate:.1f}% ({solved}/{min(total_tasks, Config.MAX_TEST_TASKS)})")

    # Save submission (partial for debug)
    output_path = "/kaggle/working/submission_debug.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(submission, f)
        print(f"\n💾 Debug submission saved to: {output_path}")
    except:
        output_path = "submission_debug.json"
        with open(output_path, 'w') as f:
            json.dump(submission, f)
        print(f"\n💾 Debug submission saved to: {output_path} (local)")

    print("\n" + "=" * 80)
    print("  🔍 DEBUG COMPLETE - Review logs above for failure analysis")
    print("=" * 80 + "\n")

    return submission

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    submission = main()
    print("\n✅ Debug execution complete. Analyze logs to identify issues! 🔍\n")
