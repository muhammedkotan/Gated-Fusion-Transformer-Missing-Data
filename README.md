# Gated Fusion Transformer for Missing Data

**Paper:** "Transformer-Based Feature Integration for Predictive Modeling in Multi-Source Business Environments"  
**Journal:** Knowledge and Information Systems (Springer) — Revision  
**Authors:** Duygu Bayrak Gümüş, Muhammed Kotan  
**Affiliation:** Sakarya University, Department of Information Systems Engineering

---

## Overview

This repository contains the complete experimental code for the **Gated Fusion Transformer**, a novel architecture for regression tasks where one data source is systematically absent at inference time.

The model is validated on a restaurant sales prediction task where engagement (view) data is structurally missing for approximately 51% of records.
---

## Repository Structure

```
├── config.py             # All constants, HPO search spaces, optimal params
├── data_processor.py     # ImprovedDataProcessor (zero-fill + NaN-preserving)
├── models.py             # All model architectures
│   ├── BaselineTransformer
│   ├── DualStreamTransformer
│   ├── AdaptiveWeightingTransformer
│   ├── GatedFusionTransformer        ← proposed model
│   └── SAINTModel                    ← external baseline 
├── training.py           # Training loops for all models
├── evaluation.py         # Statistical tests
├── hpo.py                # Hyperparameter optimisation (random search)
├── run_hpo.py            # Run HPO for all models (step 1)
├── run_phase1.py         # Phase 1 tournament (step 2)
├── run_phase2.py         # Phase 2 extended comparison (step 3)
├── requirements.txt
└── README.md
```

---

## Models

| Model | Description | Params |
|---|---|---|
| Baseline | Single-stream, all features concatenated | ~20K |
| Dual-Stream | Parallel sales/view streams, fixed 0.5 gate | ~20K |
| Adaptive | Scalar alpha weighting on view mask | ~20K |
| **GatedFusion** | **Learned soft gate (proposed)** | **~20K** |
| SAINT | Feature tokenisation + colrow attention | ~2M |
| XGBoost-Zero | XGBoost on zero-filled features | — |
| XGBoost-Native | XGBoost with NaN-preserved engagement | — |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place data file

```
restaurant_merged_data.xlsx   (sheet: Merged_Data)
```

### 3. Run experiments in order

```bash
# Step 1 — Hyperparameter optimisation (all models including SAINT)
python run_hpo.py

# Step 2 — Phase 1: Transformer ablation study (Table 2)
python run_phase1.py

# Step 3 — Phase 2: Extended comparison vs baselines (Tables 3 & 4)
python run_phase2.py
```

Results are saved to `results/`.

---

## Experimental Results

### Phase 1 — Transformer Ablation (K=5)

| Method | R² (mean ± std) | MAPE (%) |
|---|---|---|
| Baseline | 0.8073 ± 0.0698 * | 29.1 |
| Dual-Stream | 0.7317 ± 0.1470 * | 32.8 |
| Adaptive | 0.7491 ± 0.1137 * | 31.4 |
| **Gated Fusion** | **0.9161 ± 0.0177** | **27.3** |

*Significance markers vs Gated Fusion: (* p<0.05, ** p<0.01, *** p<0.001)*  
*Both paired t-test and Wilcoxon signed-rank test reported for transparency.*

### Phase 2A — General Performance (K=5)

| Method | R² (mean ± std) |
|---|---|
| **Gated Fusion** | **0.9161 ± 0.0177** |
| XGBoost-Zero | 0.9095 ± 0.0514 |
| XGBoost-Native | 0.9049 ± 0.0510 |
| SAINT | 0.7647 ± 0.0833 |

### Phase 2B — Subset Analysis (Complete vs Missing)

| Method | Complete R² | Missing R² | Δ |
|---|---|---|---|
| **Gated Fusion** | **0.9144** | **0.9256** | **+0.0111 ✓** |
| XGBoost-Zero | 0.9178 | 0.8916 | −0.0262 |
| XGBoost-Native | 0.9149 | 0.8859 | −0.0290 |
| SAINT | 0.7546 | 0.7746 | +0.0200 |

---

## Architecture: Gated Fusion Transformer

```
Sales Features ──► Sales Transformer ──► ŷ_sales ──┐
                                                      ├──► g_sales·ŷ_sales + g_full·ŷ_full
Full Features  ──► Full Transformer  ──► ŷ_full  ──┘
                                                      ↑
Category Embedding + View Flag ──► Learned Gate (Softmax)
```

The gate is conditioned on the category embedding and a binary view-availability flag. This allows the model to adapt its feature weighting at inference time without any hard-coded rule — when engagement data is absent the gate up-weights the sales stream; when it is present the gate leverages the full-feature stream.

---

## Dataset

- File: `restaurant_merged_data.xlsx`, sheet `Merged_Data`
- Shape: (6728, 26)
- Categories: 18 restaurant categories
- Missing pattern: ~51% of records lack engagement data (view_count = 0 is a genuine zero; view_duration and avg_view_duration are NaN)

---

## Reproducibility

All experiments use:
- `CV_RANDOM_STATE = 42`
- K = 5 stratified folds (stratified by `sales_main_category`)
- GPU: NVIDIA T4 (Google Colab)
- All seeds set via `set_all_seeds()` in `training.py`

---

## Citation

```bibtex
@article{bayrakgumus2025gatedfusion,
  title   = {Transformer-Based Feature Integration for Predictive Modeling
             in Multi-Source Business Environments},
  author  = {Bayrak Gümüş, Duygu and Kotan, Muhammed},
  journal = {Knowledge and Information Systems},
  year    = {2025},
  publisher = {Springer}
}
```
