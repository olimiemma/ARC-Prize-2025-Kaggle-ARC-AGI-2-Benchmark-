# 🧠 HYBRID SOTA ARC SOLVER - Production Version

*A competitive multi-tier solver for the ARC-AGI Prize 2025 competition*

## 🎯 Overview

This hybrid solver combines proven symbolic reasoning with modern LLM capabilities to tackle Abstract Reasoning Corpus (ARC) challenges. The system implements a progressive strategy that balances speed and accuracy, achieving competitive performance (15-25% expected success rate) for top leaderboard positions.

## 🏗️ Architecture

### Multi-Tier Strategy

| Tier | Method | Time Budget | Purpose |
|------|--------|-------------|---------|
| **1** | Fast Symbolic | 5 seconds | Quick palette + geometric transformations |
| **2** | Beam Search | 15 seconds | Compositional program exploration |
| **3** | LLM + TTA | 30 seconds | Complex reasoning with test-time adaptation |

### Core Components

- **`a_v4.py` DSL**: 30+ domain-specific operations for grid manipulation
- **Symbolic Solver**: Feature-driven program generation and verification
- **Beam Search**: Efficient combinatorial search with pruning
- **LLM + TTA**: Large language model with iterative feedback loops
- **Verification**: Exact match validation at every tier

## 🚀 Quick Start (Kaggle)

### Prerequisites

1. **Kaggle Account** with GPU access
2. **Required Datasets**:
   - `arc-prize-2025` (competition data)
   - `wb55L_nemomini_fulleval` (primary LLM)
   - `qwen-3` (fallback LLM)

### Setup Steps

```python
# 1. First cell in Kaggle notebook:
!pip install bitsandbytes

# 2. Create new notebook with GPU T4 x2
# 3. Add required datasets
# 4. Paste entire HYBRID_SOTA_ARC_SOLVER_FIXED.py
# 5. Run All
```

### Hardware Requirements

- **GPU**: T4 x2 (minimum)
- **RAM**: 16GB+ recommended
- **Storage**: ~10GB for models + data

## 📁 File Structure

```
/kaggle/input/
├── arc-prize-2025/                 # Competition data
│   └── arc-agi_evaluation_challenges.json
├── wb55l_nemomini_fulleval/        # Primary model
│   └── transformers/default/1/
└── qwen-3/                         # Fallback model
    └── transformers/0.6b/

/kaggle/working/
└── submission.json                  # Output predictions
```

## ⚙️ Configuration

### Key Parameters

```python
class Config:
    # Time budgets (seconds)
    TIER1_TIMEOUT = 5    # Fast symbolic
    TIER2_TIMEOUT = 15   # Beam search  
    TIER3_TIMEOUT = 30   # LLM + TTA
    
    # Search parameters
    BEAM_WIDTH = 12      # Beam search width
    BEAM_DEPTH = 3       # Maximum program depth
    
    # LLM parameters
    LLM_MAX_ATTEMPTS = 5 # TTA iterations
    LLM_MAX_TOKENS = 512
    LLM_TEMPERATURE = 0.7
```

## 🔧 DSL Operations

### Geometric Transformations
- `rotate90/180/270`, `flip_h/v`, `transpose`
- `symmetrize_h_left_to_right`, `symmetrize_v_top_to_bottom`

### Scaling & Resizing
- `tile_scale(hr, wr)` - Upscale by tiling
- `block_reduce(hr, wr)` - Downscale by majority voting
- `pad_crop_center(dh, dw, fill)` - Adjust dimensions

### Object Manipulation
- `crop_to_largest_nonbg()` - Extract main object
- `keep_largest_nonbg(fill)` - Isolate largest component
- `get_largest_nonbg_bbox()` - Bounding box detection

### Palette Operations
- `apply_palette(mapping)` - Color remapping
- Automatic palette inference from training pairs

## 🎯 Solving Strategy

### Tier 1: Fast Symbolic (5s)
- **Approach**: Feature-driven candidate generation
- **Methods**: Palette mapping, geometric transforms, size analysis
- **Use Case**: ~40-50% of solvable tasks

### Tier 2: Beam Search (15s) 
- **Approach**: Compositional program exploration
- **Methods**: Operator pool generation, beam pruning
- **Use Case**: Complex multi-step transformations

### Tier 3: LLM + TTA (30s)
- **Approach**: Language-guided program synthesis
- **Methods**: Prompt engineering, test-time adaptation
- **Use Case**: Novel patterns requiring abstract reasoning

## 📊 Performance Expectations

| Metric | Expected Range |
|--------|----------------|
| Success Rate | 15-25% |
| Tier 1 Coverage | ~40-60% of solved |
| Tier 2 Coverage | ~20-30% of solved |
| Tier 3 Coverage | ~10-20% of solved |
| Avg. Time/Task | 10-20 seconds |

## 🐛 Recent Fixes

- ✅ Fixed `'bg'` undefined variable in `keep_largest_nonbg()`
- ✅ Corrected model paths for wb55L and Qwen3
- ✅ Added `dtype=int` for grid conversions
- ✅ Fixed `torch_dtype` handling for model loading
- ✅ Improved error handling and fallbacks

## 🔍 Debugging Tips

### Common Issues

1. **Model Loading Failures**:
   - Check dataset attachments in Kaggle
   - Verify `bitsandbytes` installation
   - Use fallback model as backup

2. **Memory Issues**:
   - Reduce `BEAM_WIDTH` and `BEAM_DEPTH`
   - Limit LLM context length
   - Use smaller fallback model

3. **Timeouts**:
   - Adjust tier timeouts based on hardware
   - Disable Tier 3 if models unavailable

### Progress Monitoring

```python
# Enable detailed logging
Config.REPORT_EVERY = 10  # Report every 10 tasks

# Monitor tier performance
print(f"T1: {solver.stats['tier1_success']}, "
      f"T2: {solver.stats['tier2_success']}, "
      f"T3: {solver.stats['tier3_success']}")
```

## 🎪 Advanced Usage

### Customizing the DSL

```python
# Add custom operations to DSL class
class ExtendedDSL(DSL):
    @staticmethod
    def custom_operation(g: Grid) -> Grid:
        # Your custom logic
        return transformed_grid
```

### Modifying Search Strategy

```python
# Adjust beam search parameters
solver = HybridSolver(
    beam_width=20,    # Wider search
    depth=4,          # Deeper programs
    llm_attempts=10   # More TTA iterations
)
```

## 📈 Output Format

The solver generates submissions in the required competition format:

```json
{
  "task_id": {
    "test_case_1": {
      "attempt_1": [[...]],
      "attempt_2": [[...]]
    }
  }
}
```

## 🤝 Contributing

### Extending the Solver

1. **Add New Operations**: Extend the `DSL` class
2. **Improve Search**: Modify `BeamSearchSolver` heuristics  
3. **Enhance LLM**: Create better prompts and parsers
4. **Add Features**: Implement new feature detectors

### Performance Optimization

- Profile with `%timeit` in notebooks
- Use NumPy vectorization where possible
- Cache expensive computations
- Parallelize independent tasks

## 📄 License & Attribution

This solver integrates techniques from:
- **a_v4.py** - Proven symbolic reasoning foundation
- **wb55L Nemomini** - High-performance reasoning model
- **Qwen-3** - Robust fallback language model

## 🆘 Support

For issues:
1. Check Kaggle notebook logs
2. Verify dataset attachments
3. Ensure GPU acceleration is enabled
4. Review model compatibility

---

**Good luck with the ARC-AGI Prize 2025! 🚀**
