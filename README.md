
# ARC Prize 2025 – Kaggle Submission Notebook

This repository contains my working notebook for the [ARC Prize 2025 competition](https://www.kaggle.com/competitions/arc-prize-2025), a global challenge to advance **Artificial General Intelligence (AGI)** through the **Abstraction and Reasoning Corpus (ARC-AGI-2)** benchmark.

The competition asks participants to design algorithms capable of **novel reasoning** on grid-based puzzles. Unlike conventional supervised ML, ARC tasks are designed to be unseen and require **generalization, program synthesis, and reasoning** rather than pattern memorization.

---

## 📌 Project Overview

This notebook implements a **baseline solver** that:

* Parses **train/test pairs** from ARC task JSON files.
* Attempts a sequence of **pattern-matching strategies**:

  * Color mapping transformations
  * Grid tiling
  * Rotations and flips
  * Transpositions
* Generates **two distinct predictions per test input** (required for competition submissions).
* Falls back to **safe heuristics** (identity transform, constant fills, checkerboard diversification) when patterns cannot be detected.
* Outputs predictions in the required `submission.json` format for Kaggle.

This pipeline is intentionally simple but **robust** — it ensures valid submissions across all tasks and provides a foundation for layering in more advanced reasoning modules (DSL search, neural priors, verifiers).

---

## 🧩 Key Features

* **Pattern-Matching Heuristics**
  Detects and applies common grid transformations (rotation, reflection, color remap, tiling, transpose).

* **Per-Test Input Handling (FIXED)**
  Each test input is solved independently; no reuse of a single guess for multi-test tasks.

* **Fallback Consistency**
  If no pattern is detected, the solver produces two shape-safe outputs:

  * Attempt 1: input grid itself or padded/cropped variant.
  * Attempt 2: constant fill or checkerboard grid based on mode color of training outputs.

* **Two-Attempt Diversity**
  Always guarantees two **different** attempts per test input to maximize scoring chances.

* **Safe Runtime Practices**

  * Deterministic, reproducible generation.
  * Handles all tasks in `arc-agi_test_challenges.json`.
  * Logs warnings for fallback usage.

---

## 📂 Repository Structure

```
arc2025-notebook/
│
├── notebook.ipynb            # Kaggle-compatible notebook
├── submission.json           # Example generated submission file
├── utils/                    # Helper functions for transformations & verification
│   ├── color_mapping.py
│   ├── tiling.py
│   ├── rotations.py
│   └── verifier.py
├── README.md                 # This file
└── LICENSE                   # MIT or Apache-2.0
```

---

## 🚀 How to Run

1. **Clone repo or open in Kaggle**
   Attach the [ARC Prize 2025 dataset](https://www.kaggle.com/competitions/arc-prize-2025/data) to your notebook session.

2. **Run notebook**

   ```bash
   python notebook.ipynb
   ```

   Or simply “Commit & Run” on Kaggle.

3. **Generate submission.json**
   The notebook will automatically create a file named `submission.json` containing predictions for all tasks.

4. **Submit to Kaggle**
   Upload the `submission.json` to the competition **Submissions** tab.

---

## 🛠️ Technical Notes

* **Language/Frameworks:** Python 3, NumPy
* **No heavy dependencies** — avoids unused large libraries (`torch`, `transformers`) for faster startup.
* **Kaggle Constraints:** ≤ 12h runtime, no internet access, all external models must be mounted via Kaggle Datasets/Models.

---

## 📊 Next Steps

This baseline establishes a **valid submission pipeline**. Planned improvements include:

* Domain-Specific Language (**DSL**) for grid programs.
* Enumerative + beam search over DSL operators.
* Neural priors (small reasoning LLMs, 1–4B) for program proposals.
* Lightweight verifiers to rerank candidates.

---

## 📜 License

This project is licensed under **MIT License** (or Apache-2.0). All Kaggle competition rules apply.

---

## 🙌 Acknowledgements

* **ARC Prize Foundation** for designing and sponsoring the competition.
* **François Chollet et al.** for the original ARC dataset and benchmark.
* The Kaggle community for discussions and open baselines.

---
