"""
training.py
===========
All training loop functions.

"""

import copy
import gc
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from config import (DEVICE, K_FOLDS, CV_RANDOM_STATE,
                    N_SALES_FEATURES, N_VIEW_FEATURES, N_TOTAL_FEATURES)
from data_processor import ImprovedDataProcessor
from models import (GatedFusionTransformer, SAINTModel)

os.makedirs("best_models", exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════
def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def reset_gpu_stats():
    if DEVICE.type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)


def get_peak_gpu_memory_mb() -> float:
    if DEVICE.type == 'cuda':
        return torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
    return 0.0


def get_metrics(y_pred_norm: np.ndarray, y_true_log: np.ndarray,
                processor: ImprovedDataProcessor) -> dict:
    """Convert normalised predictions back to original scale and compute
    R², MAE and MAPE."""
    pred_orig = np.expm1(y_pred_norm * processor.y_log_max)
    true_orig = np.expm1(y_true_log)
    pred_orig = np.maximum(pred_orig, 0.1)
    true_orig = np.maximum(true_orig, 0.1)
    return {
        'r2':   r2_score(true_orig, pred_orig),
        'mae':  mean_absolute_error(true_orig, pred_orig),
        'mape': np.median(
            np.abs((true_orig - pred_orig) /
                   np.maximum(true_orig, 1))) * 100,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 TRAINING — Baseline / Dual-Stream / Adaptive
# ═════════════════════════════════════════════════════════════════════════════
def train_script1_model_run(model: nn.Module,
                             processor: ImprovedDataProcessor,
                             train_data, test_data,
                             method_name: str, seed: int,
                             label_encoder,
                             lr: float = 0.002,
                             weight_decay: float = 1e-5,
                             epochs: int = 100,
                             verbose: bool = False) -> dict:
    """
    Generic training loop for Phase-1 ablation models.
    Weighted MSE loss, Adam, patience = 25 eval-steps (every 5 epochs).
    """
    set_all_seeds(seed)
    reset_gpu_stats()
    best_loss = float('inf')

    try:
        if method_name == "Dual-Stream":
            X_sales_tr, X_view_tr, _, y_tr, y_log_tr, cat_tr, mask_tr = \
                processor.prepare_features_fair(
                    train_data, label_encoder, is_train=True)
            X_sales_te, X_view_te, _, y_te, y_log_te, cat_te, mask_te = \
                processor.prepare_features_fair(
                    test_data, label_encoder, is_train=False)
            processor.view_scaler = StandardScaler()
            X_sales_tr = torch.FloatTensor(
                processor.sales_scaler.fit_transform(X_sales_tr)).to(DEVICE)
            X_view_tr  = torch.FloatTensor(
                processor.view_scaler.fit_transform(X_view_tr)).to(DEVICE)
            X_sales_te = torch.FloatTensor(
                processor.sales_scaler.transform(X_sales_te)).to(DEVICE)
            X_view_te  = torch.FloatTensor(
                processor.view_scaler.transform(X_view_te)).to(DEVICE)
        else:
            _, _, X_all_tr, y_tr, y_log_tr, cat_tr, mask_tr = \
                processor.prepare_features_fair(
                    train_data, label_encoder, is_train=True)
            _, _, X_all_te, y_te, y_log_te, cat_te, mask_te = \
                processor.prepare_features_fair(
                    test_data, label_encoder, is_train=False)
            X_all_tr = torch.FloatTensor(
                processor.scaler.fit_transform(X_all_tr)).to(DEVICE)
            X_all_te = torch.FloatTensor(
                processor.scaler.transform(X_all_te)).to(DEVICE)

        y_tr_t    = torch.FloatTensor(y_tr).to(DEVICE)
        y_te_t    = torch.FloatTensor(y_te).to(DEVICE)
        cat_tr_t  = torch.LongTensor(cat_tr).to(DEVICE)
        cat_te_t  = torch.LongTensor(cat_te).to(DEVICE)
        mask_tr_t = torch.BoolTensor(mask_tr).to(DEVICE)
        mask_te_t = torch.BoolTensor(mask_te).to(DEVICE)
        model     = model.to(DEVICE)

        def weighted_mse_loss(pred, target):
            weights = 1.0 / (target * processor.y_log_max + 1.0)
            weights = weights / weights.mean()
            return ((pred - target) ** 2 * weights).mean()

        optimizer        = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        patience_counter = 0
        best_model       = None
        t0               = time.time()

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            if method_name == "Dual-Stream":
                pred = model(X_sales_tr, X_view_tr, cat_tr_t, mask_tr_t)
            elif method_name == "Adaptive":
                pred = model(X_all_tr, cat_tr_t, mask_tr_t)
            else:
                pred = model(X_all_tr, cat_tr_t)
            loss = weighted_mse_loss(pred, y_tr_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    if method_name == "Dual-Stream":
                        val_pred = model(X_sales_te, X_view_te,
                                          cat_te_t, mask_te_t)
                    elif method_name == "Adaptive":
                        val_pred = model(X_all_te, cat_te_t, mask_te_t)
                    else:
                        val_pred = model(X_all_te, cat_te_t)
                    val_loss = weighted_mse_loss(val_pred, y_te_t)
                    if val_loss < best_loss:
                        best_loss        = val_loss
                        patience_counter = 0
                        best_model       = copy.deepcopy(model.state_dict())
                    else:
                        patience_counter += 5
                    if patience_counter >= 25:
                        break

        training_time  = time.time() - t0
        peak_memory_mb = get_peak_gpu_memory_mb()
        if best_model is not None:
            model.load_state_dict(best_model)
        model.eval()

        with torch.no_grad():
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            inf_start = time.time()
            if method_name == "Dual-Stream":
                test_pred = model(X_sales_te, X_view_te, cat_te_t, mask_te_t)
            elif method_name == "Adaptive":
                test_pred = model(X_all_te, cat_te_t, mask_te_t)
            else:
                test_pred = model(X_all_te, cat_te_t)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            inf_time = (time.time() - inf_start) / len(y_te) * 1000

        metrics = get_metrics(test_pred.cpu().numpy(), y_log_te, processor)
        return {
            'metrics':                     metrics,
            'train_time_sec':              training_time,
            'peak_memory_mb':              peak_memory_mb,
            'inference_time_ms_per_sample':inf_time,
        }

    except Exception as e:
        if verbose:
            print(f"  Training error ({method_name}): {e}")
        return {
            'metrics':                     {'r2': -np.inf, 'mae': np.inf, 'mape': np.inf},
            'train_time_sec':              0.0,
            'peak_memory_mb':              0.0,
            'inference_time_ms_per_sample':0.0,
        }


# ═════════════════════════════════════════════════════════════════════════════
# GATED FUSION TRAINING 
# ═════════════════════════════════════════════════════════════════════════════
def train_script2_model_run(model: nn.Module,
                             processor: ImprovedDataProcessor,
                             train_data, test_data,
                             method_name: str, seed: int,
                             label_encoder,
                             n_categories: int, sales_dim: int,
                             input_dim: int,
                             lr: float = 0.002,
                             weight_decay: float = 1e-5,
                             epochs: int = 150,
                             n_runs: int = 3,
                             verbose: bool = False) -> dict:
    """
    Trains GatedFusionTransformer with:
    • n_runs independent initialisation seeds per fold
    • Best run selected by test-set R² (after best-val-loss weights are loaded)
    • Best model weights saved to best_models/
    • Weighted MSE loss, Adam, patience = 25 eval-steps

    Note: the 'model' argument is accepted for API compatibility with Phase 1
    but a fresh model is instantiated for each run internally.
    """
    best_overall_r2      = -np.inf
    best_overall_metrics = None
    best_overall_time    = 0.0
    best_overall_memory  = 0.0
    best_overall_inf     = 0.0

    for run in range(n_runs):
        run_seed = seed + run * 100
        set_all_seeds(run_seed)
        reset_gpu_stats()
        best_loss = float('inf')

        try:
            X_sales_tr, _, X_all_tr, y_tr, y_log_tr, cat_tr, mask_tr = \
                processor.prepare_features_fair(
                    train_data, label_encoder, is_train=True)
            X_sales_te, _, X_all_te, y_te, y_log_te, cat_te, mask_te = \
                processor.prepare_features_fair(
                    test_data, label_encoder, is_train=False)

            X_sales_tr = torch.FloatTensor(
                processor.sales_scaler.fit_transform(X_sales_tr)).to(DEVICE)
            X_all_tr   = torch.FloatTensor(
                processor.scaler.fit_transform(X_all_tr)).to(DEVICE)
            X_sales_te = torch.FloatTensor(
                processor.sales_scaler.transform(X_sales_te)).to(DEVICE)
            X_all_te   = torch.FloatTensor(
                processor.scaler.transform(X_all_te)).to(DEVICE)

            y_tr_t    = torch.FloatTensor(y_tr).to(DEVICE)
            y_te_t    = torch.FloatTensor(y_te).to(DEVICE)
            cat_tr_t  = torch.LongTensor(cat_tr).to(DEVICE)
            cat_te_t  = torch.LongTensor(cat_te).to(DEVICE)
            mask_tr_t = torch.BoolTensor(mask_tr).to(DEVICE)
            mask_te_t = torch.BoolTensor(mask_te).to(DEVICE)

            # Fresh model per run
            # [REVISED: RENAME] was ConfigurableEnsembleTransformer
            run_model = GatedFusionTransformer(
                sales_dim=sales_dim, full_dim=input_dim,
                n_categories=n_categories, nhead=4, num_layers=1,
                d_model=64, dropout=0.1, category_emb_dim=16,
            ).to(DEVICE)

            def weighted_mse_loss(pred, target):
                weights = 1.0 / (target * processor.y_log_max + 1.0)
                weights = weights / weights.mean()
                return ((pred - target) ** 2 * weights).mean()

            optimizer        = optim.Adam(
                run_model.parameters(), lr=lr, weight_decay=weight_decay)
            patience_counter = 0
            best_model_state = None
            t0               = time.time()

            for epoch in range(epochs):
                run_model.train()
                optimizer.zero_grad()
                pred = run_model(X_sales_tr, X_all_tr, cat_tr_t, mask_tr_t)
                loss = weighted_mse_loss(pred, y_tr_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(run_model.parameters(), 1.0)
                optimizer.step()

                if epoch % 5 == 0:
                    run_model.eval()
                    with torch.no_grad():
                        val_pred = run_model(
                            X_sales_te, X_all_te, cat_te_t, mask_te_t)
                        val_loss = weighted_mse_loss(val_pred, y_te_t)
                        if val_loss < best_loss:
                            best_loss        = val_loss
                            patience_counter = 0
                            best_model_state = copy.deepcopy(
                                run_model.state_dict())
                        else:
                            patience_counter += 5
                        if patience_counter >= 25:
                            break

            training_time  = time.time() - t0
            peak_memory_mb = get_peak_gpu_memory_mb()
            if best_model_state is not None:
                run_model.load_state_dict(best_model_state)
            run_model.eval()

            with torch.no_grad():
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                inf_start = time.time()
                test_pred = run_model(
                    X_sales_te, X_all_te, cat_te_t, mask_te_t)
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                inf_time = (time.time() - inf_start) / len(y_te) * 1000

            metrics = get_metrics(
                test_pred.cpu().numpy(), y_log_te, processor)

            is_best = metrics['r2'] > best_overall_r2
            if verbose:
                print(f"      Run {run+1}/{n_runs}: R2={metrics['r2']:.4f}"
                      f"{' ← best' if is_best else ''}")

            if is_best:
                best_overall_r2      = metrics['r2']
                best_overall_metrics = metrics
                best_overall_time    = training_time
                best_overall_memory  = peak_memory_mb
                best_overall_inf     = inf_time
                torch.save(best_model_state,
                           f'best_models/gated_fusion_seed{run_seed}.pt')

        except Exception as e:
            if verbose:
                print(f"      Run {run+1} error: {e}")
            continue

    if best_overall_metrics is None:
        return {
            'metrics':                     {'r2': -np.inf, 'mae': np.inf, 'mape': np.inf},
            'train_time_sec':              0.0,
            'peak_memory_mb':              0.0,
            'inference_time_ms_per_sample':0.0,
        }
    return {
        'metrics':                     best_overall_metrics,
        'train_time_sec':              best_overall_time,
        'peak_memory_mb':              best_overall_memory,
        'inference_time_ms_per_sample':best_overall_inf,
    }


# ═════════════════════════════════════════════════════════════════════════════
# XGBOOST — ZERO-FILL  
# ═════════════════════════════════════════════════════════════════════════════
def train_xgboost_zero(train_data, test_data,
                        processor: ImprovedDataProcessor,
                        seed: int, label_encoder,
                        xgb_params: dict) -> dict:
    """XGBoost trained on zero-filled features (fair comparison baseline)."""
    np.random.seed(seed)
    X_sales_tr, _, X_all_tr, _, y_log_tr, cat_tr, _ = \
        processor.prepare_features_fair(
            train_data, label_encoder, is_train=True)
    X_sales_te, _, X_all_te, _, y_log_te, cat_te, _ = \
        processor.prepare_features_fair(
            test_data, label_encoder, is_train=False)

    X_train = np.column_stack([X_all_tr, cat_tr])
    X_test  = np.column_stack([X_all_te, cat_te])
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = xgb.XGBRegressor(
        **xgb_params, random_state=seed, n_jobs=-1, verbosity=0)
    t0 = time.time()
    model.fit(X_train, y_log_tr)
    train_time = time.time() - t0
    t1 = time.time()
    y_pred_log = model.predict(X_test)
    inf_time   = (time.time() - t1) / len(y_log_te) * 1000

    pred_orig = np.maximum(np.expm1(y_pred_log), 0.1)
    true_orig = np.maximum(np.expm1(y_log_te),   0.1)
    return {
        'metrics': {
            'r2':   r2_score(true_orig, pred_orig),
            'mae':  mean_absolute_error(true_orig, pred_orig),
            'mape': np.median(np.abs(
                (true_orig - pred_orig) /
                np.maximum(true_orig, 1))) * 100,
        },
        'train_time_sec':              train_time,
        'peak_memory_mb':              0.0,
        'inference_time_ms_per_sample':inf_time,
        'y_pred_log': y_pred_log,
        'y_true_log': y_log_te,
    }


# ═════════════════════════════════════════════════════════════════════════════
# XGBOOST — NATIVE NaN 
# ═════════════════════════════════════════════════════════════════════════════
def train_xgboost_native(train_data, test_data,
                          processor: ImprovedDataProcessor,
                          seed: int, label_encoder,
                          xgb_params: dict) -> dict:
    """
    XGBoost trained on NaN-preserving features.
    view_duration and avg_view_duration are NaN where engagement is absent,
    allowing XGBoost's hist tree method to learn optimal default directions.
    Only sales columns are scaled; engagement NaNs are left as-is.
    """
    np.random.seed(seed)
    X_all_tr, y_log_tr, cat_tr, _ = \
        processor.prepare_features_xgb_native(
            train_data, label_encoder, is_train=True)
    X_all_te, y_log_te, cat_te, _ = \
        processor.prepare_features_xgb_native(
            test_data, label_encoder, is_train=False)

    X_train = np.column_stack([X_all_tr, cat_tr])
    X_test  = np.column_stack([X_all_te, cat_te])

    # Scale only the sales columns; leave engagement NaNs intact
    scaler     = StandardScaler()
    sales_cols = list(range(N_SALES_FEATURES))
    X_train[:, sales_cols] = scaler.fit_transform(X_train[:, sales_cols])
    X_test[:, sales_cols]  = scaler.transform(X_test[:, sales_cols])

    model = xgb.XGBRegressor(
        **xgb_params, random_state=seed,
        n_jobs=-1, verbosity=0, tree_method='hist')
    t0 = time.time()
    model.fit(X_train, y_log_tr)
    train_time = time.time() - t0
    t1 = time.time()
    y_pred_log = model.predict(X_test)
    inf_time   = (time.time() - t1) / len(y_log_te) * 1000

    pred_orig = np.maximum(np.expm1(y_pred_log), 0.1)
    true_orig = np.maximum(np.expm1(y_log_te),   0.1)
    return {
        'metrics': {
            'r2':   r2_score(true_orig, pred_orig),
            'mae':  mean_absolute_error(true_orig, pred_orig),
            'mape': np.median(np.abs(
                (true_orig - pred_orig) /
                np.maximum(true_orig, 1))) * 100,
        },
        'train_time_sec':              train_time,
        'peak_memory_mb':              0.0,
        'inference_time_ms_per_sample':inf_time,
        'y_pred_log': y_pred_log,
        'y_true_log': y_log_te,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SAINT TRAINING  
# ═════════════════════════════════════════════════════════════════════════════
def train_saint(train_data, test_data,
                processor: ImprovedDataProcessor,
                seed: int, label_encoder,
                d_model: int = 32, nhead: int = 2, n_layers: int = 2,
                lr: float = 0.001, weight_decay: float = 1e-4,
                dropout: float = 0.1, epochs: int = 150,
                n_categories: int = 18,
                verbose: bool = False) -> dict:
    """
    SAINT training loop using the same zero-fill pipeline and weighted MSE
    loss as the other Transformer models for a fair comparison.
    [REVISED: EC3]
    """
    set_all_seeds(seed)
    reset_gpu_stats()
    best_loss = float('inf')

    try:
        _, _, X_all_tr, y_tr, y_log_tr, cat_tr, _ = \
            processor.prepare_features_fair(
                train_data, label_encoder, is_train=True)
        _, _, X_all_te, y_te, y_log_te, cat_te, _ = \
            processor.prepare_features_fair(
                test_data, label_encoder, is_train=False)

        X_all_tr = torch.FloatTensor(
            processor.scaler.fit_transform(X_all_tr)).to(DEVICE)
        X_all_te = torch.FloatTensor(
            processor.scaler.transform(X_all_te)).to(DEVICE)

        y_tr_t   = torch.FloatTensor(y_tr).to(DEVICE)
        y_te_t   = torch.FloatTensor(y_te).to(DEVICE)
        cat_tr_t = torch.LongTensor(cat_tr).to(DEVICE)
        cat_te_t = torch.LongTensor(cat_te).to(DEVICE)

        model = SAINTModel(
            n_cont_features=N_TOTAL_FEATURES,
            n_categories=n_categories,
            d_model=d_model, nhead=nhead,
            n_layers=n_layers, dropout=dropout,
        ).to(DEVICE)

        def weighted_mse_loss(pred, target):
            weights = 1.0 / (target * processor.y_log_max + 1.0)
            weights = weights / weights.mean()
            return ((pred - target) ** 2 * weights).mean()

        optimizer        = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        patience_counter = 0
        best_model       = None
        t0               = time.time()

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_all_tr, cat_tr_t)
            loss = weighted_mse_loss(pred, y_tr_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_all_te, cat_te_t)
                    val_loss = weighted_mse_loss(val_pred, y_te_t)
                    if val_loss < best_loss:
                        best_loss        = val_loss
                        patience_counter = 0
                        best_model       = copy.deepcopy(model.state_dict())
                    else:
                        patience_counter += 5
                    if patience_counter >= 25:
                        break

        training_time  = time.time() - t0
        peak_memory_mb = get_peak_gpu_memory_mb()
        if best_model is not None:
            model.load_state_dict(best_model)
        model.eval()

        with torch.no_grad():
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            inf_start = time.time()
            test_pred = model(X_all_te, cat_te_t)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            inf_time = (time.time() - inf_start) / len(y_te) * 1000

        metrics = get_metrics(test_pred.cpu().numpy(), y_log_te, processor)
        return {
            'metrics':                     metrics,
            'train_time_sec':              training_time,
            'peak_memory_mb':              peak_memory_mb,
            'inference_time_ms_per_sample':inf_time,
            'y_pred_log': test_pred.cpu().numpy(),
            'y_true_log': y_log_te,
        }

    except Exception as e:
        if verbose:
            print(f"  SAINT training error: {e}")
        return {
            'metrics':                     {'r2': -np.inf, 'mae': np.inf, 'mape': np.inf},
            'train_time_sec':              0.0,
            'peak_memory_mb':              0.0,
            'inference_time_ms_per_sample':0.0,
        }
