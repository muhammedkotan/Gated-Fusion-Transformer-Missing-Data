## Overview

This repository contains the complete experimental code for the **Gated Fusion Transformer**, an architecture for regression tasks where one data source is systematically absent at inference time.

The model is validated on a restaurant sales prediction task where engagement (view) data is structurally missing for approximately half of the records.
---

## Models

| Model | Description | 
|---|---|
| Baseline | Single-stream, all features concatenated | 
| Dual-Stream | Parallel sales/view streams, fixed 0.5 gate | 
| Adaptive | Scalar alpha weighting on view mask | 
| **GatedFusion** | **Learned soft gate ** | 



## Architecture: Gated Fusion Transformer

```
Sales Features ──► Sales Transformer ──► ŷ_sales ──┐
                                                      ├──► g_sales·ŷ_sales + g_full·ŷ_full
Full Features  ──► Full Transformer  ──► ŷ_full  ──┘
                                                      ↑
Category Embedding + View Flag ──► Learned Gate (Softmax)
```



---

## Reproducibility

All experiments use:
- `CV_RANDOM_STATE = 42`
- K = 5 stratified folds (stratified by `sales_main_category`)
- GPU: NVIDIA A100 

---

## Citation

```bibtex
@article{kotan2026gatedfusion,
  title   = {Transformer-Based Feature Integration for Predictive Modeling
             in Multi-Source Business Environments},
  author  = {Bayrak Gümüş, Duygu and Kotan, Muhammed},
  journal = {},
  year    = {2026},
  publisher = {}
}
```
