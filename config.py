"""
config.py
=========
Central configuration for the Gated Fusion Transformer project.
All constants, HPO search spaces, and best-found hyperparameters
live here.  Edit this file before running any pipeline script.
"""

import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = "restaurant_merged_data.xlsx"
SHEET_NAME   = "Merged_Data"
RESULTS_DIR  = Path("results")
MODELS_DIR   = Path("best_models")

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
K_FOLDS         = 5
CV_RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DIMENSIONS  (fixed by dataset — do not change)
# ─────────────────────────────────────────────────────────────────────────────
N_SALES_FEATURES  = 5   
N_VIEW_FEATURES   = 5   
N_TOTAL_FEATURES  = N_SALES_FEATURES + N_VIEW_FEATURES   # 10

TARGET_COL   = "net_sales_qty"
CATEGORY_COL = "sales_main_category"
N_CATEGORIES_DEFAULT = 18   

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMAL HYPERPARAMETERS  (best found after Round-3 HPO run)
# ─────────────────────────────────────────────────────────────────────────────
OPTIMAL_PARAMS = {
    "Baseline": {
        "lr": 0.005,
        "d_model": 32,
        "dropout": 0.1,
        "weight_decay": 1e-5,
    },
    "Dual-Stream": {
        "lr": 0.005,
        "d_model": 128,
        "dropout": 0.1,
        "weight_decay": 1e-5,
    },
    "Adaptive": {
        "lr": 0.005,
        "d_model": 32,
        "dropout": 0.1,
        "weight_decay": 1e-5,
        "alpha": 0.3,
    },
    # [REVISED: RENAME] was "ConfigurableEnsemble"
    "GatedFusion": {
        "lr": 0.003,
        "d_model": 128,
        "dropout": 0.1,
        "weight_decay": 1e-5,
        "category_emb_dim": 8,
    },
    "XGBoost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
    },
    "SAINT": {
        "d_model": 32,
        "nhead": 2,
        "n_layers": 2,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "epochs": 150,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# HPO SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
N_ITER_HPO = 30   # random-search trials per model

HPO_SEARCH_SPACES = {
    "Baseline": {
        "lr":           [0.001, 0.005, 0.01, 0.02],
        "d_model":      [32, 64, 128],
        "dropout":      [0.0, 0.1, 0.2, 0.3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    "Dual-Stream": {
        "lr":           [0.001, 0.005, 0.01, 0.02],
        "d_model":      [32, 64, 128],
        "dropout":      [0.0, 0.1, 0.2, 0.3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
    "Adaptive": {
        "lr":           [0.001, 0.005, 0.01, 0.02],
        "d_model":      [32, 64, 128],
        "dropout":      [0.0, 0.1, 0.2, 0.3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
        "alpha":        [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "GatedFusion": {
        "lr":               [0.001, 0.003, 0.005, 0.01],
        "d_model":          [32, 64, 128],
        "dropout":          [0.0, 0.1, 0.2, 0.3],
        "weight_decay":     [1e-5, 1e-4, 1e-3],
        "category_emb_dim": [8, 16, 32],
    },
    "XGBoost": {
        "n_estimators":     [50, 100, 200, 300],
        "max_depth":        [3, 4, 5, 6, 7],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "SAINT": {
        "d_model":      [32, 64, 128],
        "nhead":        [2, 4],
        "n_layers":     [1, 2, 3],
        "lr":           [0.0001, 0.0005, 0.001, 0.005],
        "dropout":      [0.0, 0.1, 0.2, 0.3],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    },
}
