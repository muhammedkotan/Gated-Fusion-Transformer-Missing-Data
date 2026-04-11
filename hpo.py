"""
hpo.py
======
Hyperparameter optimisation (random search) for all models.
"""

import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import (DEVICE, CV_RANDOM_STATE, N_VIEW_FEATURES,
                    N_TOTAL_FEATURES, N_ITER_HPO, HPO_SEARCH_SPACES,
                    RESULTS_DIR)
from data_processor import ImprovedDataProcessor
from models import (BaselineTransformer, DualStreamTransformer,
                    AdaptiveWeightingTransformer, GatedFusionTransformer,
                    SAINTModel)
from training import set_all_seeds, get_metrics


# ═════════════════════════════════════════════════════════════════════════════
# TRANSFORMER HPO TRIAL
# ═════════════════════════════════════════════════════════════════════════════
def run_hpo_trial_transformer(model_name: str, params: dict,
                               train_data, val_data,
                               label_encoder, n_categories: int,
                               sales_dim: int, input_dim: int,
                               seed: int = 42) -> float:
    """
    Trains one Transformer model with given params on train_data,
    evaluates on val_data.  Returns best validation R².
    """
    set_all_seeds(seed)
    processor = ImprovedDataProcessor()

    try:
        # Feature preparation per model type
        if model_name == 'Dual-Stream':
            X_sales_tr, X_view_tr, _, y_tr, y_log_tr, cat_tr, mask_tr = \
                processor.prepare_features_fair(
                    train_data, label_encoder, is_train=True)
            X_sales_vl, X_view_vl, _, y_vl, y_log_vl, cat_vl, mask_vl = \
                processor.prepare_features_fair(
                    val_data, label_encoder, is_train=False)
            processor.view_scaler = StandardScaler()
            X_sales_tr = torch.FloatTensor(
                processor.sales_scaler.fit_transform(X_sales_tr)).to(DEVICE)
            X_view_tr  = torch.FloatTensor(
                processor.view_scaler.fit_transform(X_view_tr)).to(DEVICE)
            X_sales_vl = torch.FloatTensor(
                processor.sales_scaler.transform(X_sales_vl)).to(DEVICE)
            X_view_vl  = torch.FloatTensor(
                processor.view_scaler.transform(X_view_vl)).to(DEVICE)

        # [REVISED: RENAME] was 'ConfigurableEnsemble'
        elif model_name == 'GatedFusion':
            X_sales_tr, _, X_all_tr, y_tr, y_log_tr, cat_tr, mask_tr = \
                processor.prepare_features_fair(
                    train_data, label_encoder, is_train=True)
            X_sales_vl, _, X_all_vl, y_vl, y_log_vl, cat_vl, mask_vl = \
                processor.prepare_features_fair(
                    val_data, label_encoder, is_train=False)
            X_sales_tr = torch.FloatTensor(
                processor.sales_scaler.fit_transform(X_sales_tr)).to(DEVICE)
            X_all_tr   = torch.FloatTensor(
                processor.scaler.fit_transform(X_all_tr)).to(DEVICE)
            X_sales_vl = torch.FloatTensor(
                processor.sales_scaler.transform(X_sales_vl)).to(DEVICE)
            X_all_vl   = torch.FloatTensor(
                processor.scaler.transform(X_all_vl)).to(DEVICE)

        else:   # Baseline, Adaptive
            _, _, X_all_tr, y_tr, y_log_tr, cat_tr, mask_tr = \
                processor.prepare_features_fair(
                    train_data, label_encoder, is_train=True)
            _, _, X_all_vl, y_vl, y_log_vl, cat_vl, mask_vl = \
                processor.prepare_features_fair(
                    val_data, label_encoder, is_train=False)
            X_all_tr = torch.FloatTensor(
                processor.scaler.fit_transform(X_all_tr)).to(DEVICE)
            X_all_vl = torch.FloatTensor(
                processor.scaler.transform(X_all_vl)).to(DEVICE)

        y_tr_t    = torch.FloatTensor(y_tr).to(DEVICE)
        y_vl_t    = torch.FloatTensor(y_vl).to(DEVICE)
        cat_tr_t  = torch.LongTensor(cat_tr).to(DEVICE)
        cat_vl_t  = torch.LongTensor(cat_vl).to(DEVICE)
        mask_tr_t = torch.BoolTensor(mask_tr).to(DEVICE)
        mask_vl_t = torch.BoolTensor(mask_vl).to(DEVICE)

        d_model = params.get('d_model', 64)
        dropout = params.get('dropout', 0.1)

        # Model instantiation
        if model_name == 'Baseline':
            model = BaselineTransformer(
                input_dim=input_dim, d_model=d_model, nhead=4,
                n_categories=n_categories, dropout=dropout).to(DEVICE)
        elif model_name == 'Dual-Stream':
            model = DualStreamTransformer(
                sales_dim=sales_dim, view_dim=N_VIEW_FEATURES,
                d_model=d_model, nhead=4,
                n_categories=n_categories, dropout=dropout).to(DEVICE)
        elif model_name == 'Adaptive':
            model = AdaptiveWeightingTransformer(
                input_dim=input_dim, d_model=d_model, nhead=4,
                n_categories=n_categories, dropout=dropout,
                alpha=params.get('alpha', 0.5)).to(DEVICE)
        # [REVISED: RENAME] was 'ConfigurableEnsemble'
        elif model_name == 'GatedFusion':
            model = GatedFusionTransformer(
                sales_dim=sales_dim, full_dim=input_dim,
                d_model=d_model, nhead=4, num_layers=1,
                n_categories=n_categories, dropout=dropout,
                category_emb_dim=params.get(
                    'category_emb_dim', 16)).to(DEVICE)
        else:
            return -np.inf

        def weighted_mse_loss(pred, target):
            weights = 1.0 / (target * processor.y_log_max + 1.0)
            weights = weights / weights.mean()
            return ((pred - target) ** 2 * weights).mean()

        optimizer        = optim.Adam(
            model.parameters(),
            lr=params['lr'], weight_decay=params['weight_decay'])
        best_val_r2      = -np.inf
        patience_counter = 0

        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            if model_name == 'Dual-Stream':
                pred = model(X_sales_tr, X_view_tr, cat_tr_t, mask_tr_t)
            elif model_name == 'Adaptive':
                pred = model(X_all_tr, cat_tr_t, mask_tr_t)
            elif model_name == 'GatedFusion':
                pred = model(X_sales_tr, X_all_tr, cat_tr_t, mask_tr_t)
            else:
                pred = model(X_all_tr, cat_tr_t)
            loss = weighted_mse_loss(pred, y_tr_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    if model_name == 'Dual-Stream':
                        val_pred = model(X_sales_vl, X_view_vl,
                                          cat_vl_t, mask_vl_t)
                    elif model_name == 'Adaptive':
                        val_pred = model(X_all_vl, cat_vl_t, mask_vl_t)
                    elif model_name == 'GatedFusion':
                        val_pred = model(X_sales_vl, X_all_vl,
                                          cat_vl_t, mask_vl_t)
                    else:
                        val_pred = model(X_all_vl, cat_vl_t)
                    val_r2 = get_metrics(
                        val_pred.cpu().numpy(), y_log_vl, processor)['r2']
                    if val_r2 > best_val_r2:
                        best_val_r2      = val_r2
                        patience_counter = 0
                    else:
                        patience_counter += 5
                    if patience_counter >= 25:
                        break

        return best_val_r2

    except Exception:
        return -np.inf


# ═════════════════════════════════════════════════════════════════════════════
# XGBOOST HPO TRIAL
# ═════════════════════════════════════════════════════════════════════════════
def run_hpo_trial_xgboost(params: dict, train_data, val_data,
                           label_encoder, seed: int = 42) -> float:
    np.random.seed(seed)
    processor = ImprovedDataProcessor()

    _, _, X_all_tr, _, y_log_tr, cat_tr, _ = \
        processor.prepare_features_fair(
            train_data, label_encoder, is_train=True)
    _, _, X_all_vl, _, y_log_vl, cat_vl, _ = \
        processor.prepare_features_fair(
            val_data, label_encoder, is_train=False)

    X_train = np.column_stack([X_all_tr, cat_tr])
    X_val   = np.column_stack([X_all_vl, cat_vl])
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    model = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        random_state=seed, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_log_tr)
    y_pred = model.predict(X_val)

    pred_orig = np.maximum(np.expm1(y_pred),     0.1)
    true_orig = np.maximum(np.expm1(y_log_vl),   0.1)
    return r2_score(true_orig, pred_orig)


# ═════════════════════════════════════════════════════════════════════════════
# SAINT HPO TRIAL 
# ═════════════════════════════════════════════════════════════════════════════
def run_hpo_trial_saint(params: dict, train_data, val_data,
                         label_encoder, n_categories: int,
                         seed: int = 42) -> float:
    """
    Single SAINT HPO trial.
    Uses zero-fill preprocessing (identical to Phase 1/2 SAINT training)
    for a fair hyperparameter search.
    [REVISED: EC3]
    """
    set_all_seeds(seed)
    processor = ImprovedDataProcessor()

    try:
        _, _, X_all_tr, y_tr, y_log_tr, cat_tr, _ = \
            processor.prepare_features_fair(
                train_data, label_encoder, is_train=True)
        _, _, X_all_vl, y_vl, y_log_vl, cat_vl, _ = \
            processor.prepare_features_fair(
                val_data, label_encoder, is_train=False)

        X_all_tr = torch.FloatTensor(
            processor.scaler.fit_transform(X_all_tr)).to(DEVICE)
        X_all_vl = torch.FloatTensor(
            processor.scaler.transform(X_all_vl)).to(DEVICE)

        y_tr_t   = torch.FloatTensor(y_tr).to(DEVICE)
        y_vl_t   = torch.FloatTensor(y_vl).to(DEVICE)
        cat_tr_t = torch.LongTensor(cat_tr).to(DEVICE)
        cat_vl_t = torch.LongTensor(cat_vl).to(DEVICE)

        model = SAINTModel(
            n_cont_features=N_TOTAL_FEATURES,
            n_categories=n_categories,
            d_model=params.get('d_model', 32),
            nhead=params.get('nhead', 2),
            n_layers=params.get('n_layers', 2),
            dropout=params.get('dropout', 0.1),
        ).to(DEVICE)

        def weighted_mse_loss(pred, target):
            weights = 1.0 / (target * processor.y_log_max + 1.0)
            weights = weights / weights.mean()
            return ((pred - target) ** 2 * weights).mean()

        optimizer        = optim.Adam(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params.get('weight_decay', 1e-4))
        best_val_r2      = -np.inf
        patience_counter = 0

        for epoch in range(100):
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
                    val_pred = model(X_all_vl, cat_vl_t)
                    val_r2   = get_metrics(
                        val_pred.cpu().numpy(), y_log_vl, processor)['r2']
                    if val_r2 > best_val_r2:
                        best_val_r2      = val_r2
                        patience_counter = 0
                    else:
                        patience_counter += 5
                    if patience_counter >= 25:
                        break

        return best_val_r2

    except Exception as e:
        return -np.inf


# ═════════════════════════════════════════════════════════════════════════════
# MAIN HPO RUNNER
# ═════════════════════════════════════════════════════════════════════════════
def run_hpo(full_data, label_encoder, n_categories: int,
            sales_dim: int, input_dim: int,
            stratify_groups,
            n_iter: int = N_ITER_HPO) -> dict:
    """
    Runs randomised hyperparameter search for all models.
    Uses the first K-Fold split (80% train / 20% validation).

    Returns
    -------
    best_params_all : dict  — {model_name: best_params_dict}
    Also saves results to results/hpo_optimal_params.json
    """
    print("=" * 70)
    print("HYPERPARAMETER OPTIMIZATION")
    print(f"Search iterations per model: {n_iter}")
    print("=" * 70)

    kfold      = StratifiedKFold(n_splits=5, shuffle=True,
                                  random_state=CV_RANDOM_STATE)
    splits     = list(kfold.split(full_data, stratify_groups))
    train_idx, val_idx = splits[0]
    train_data = full_data.iloc[train_idx]
    val_data   = full_data.iloc[val_idx]
    print(f"\nHPO Split: {len(train_data)} train, "
          f"{len(val_data)} validation samples")

    best_params_all = {}

    # ── Transformer models ─────────────────────────────────────────────────
    transformer_models = ['Baseline', 'Dual-Stream', 'Adaptive', 'GatedFusion']

    for model_name in transformer_models:
        print(f"\n{'─'*50}")
        print(f"Optimizing: {model_name}")
        print(f"{'─'*50}")

        search_space = HPO_SEARCH_SPACES[model_name]
        best_r2      = -np.inf
        best_params  = None

        for trial in range(n_iter):
            trial_params = {
                k: random.choice(v)
                for k, v in search_space.items()}

            val_r2 = run_hpo_trial_transformer(
                model_name, trial_params,
                train_data, val_data,
                label_encoder, n_categories,
                sales_dim, input_dim,
                seed=CV_RANDOM_STATE + trial)

            if val_r2 > best_r2:
                best_r2     = val_r2
                best_params = trial_params.copy()
                print(f"  Trial {trial+1:>2}/{n_iter}: "
                      f"R2={val_r2:.4f} ← new best {trial_params}")
            else:
                print(f"  Trial {trial+1:>2}/{n_iter}: R2={val_r2:.4f}")

        best_params_all[model_name] = best_params
        print(f"\n  Best params for {model_name}: {best_params}")
        print(f"  Best val R2: {best_r2:.4f}")

    # ── XGBoost ────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Optimizing: XGBoost")
    print(f"{'─'*50}")
    best_r2 = -np.inf; best_params = None
    for trial in range(n_iter):
        trial_params = {k: random.choice(v)
                        for k, v in HPO_SEARCH_SPACES['XGBoost'].items()}
        val_r2 = run_hpo_trial_xgboost(
            trial_params, train_data, val_data, label_encoder,
            seed=CV_RANDOM_STATE + trial)
        if val_r2 > best_r2:
            best_r2     = val_r2
            best_params = trial_params.copy()
            print(f"  Trial {trial+1:>2}/{n_iter}: "
                  f"R2={val_r2:.4f} ← new best {trial_params}")
        else:
            print(f"  Trial {trial+1:>2}/{n_iter}: R2={val_r2:.4f}")
    best_params_all['XGBoost'] = best_params
    print(f"\n  Best params for XGBoost: {best_params}")

    # ── SAINT ─────────────────────────────────
    print(f"\n{'─'*50}")
    print("Optimizing: SAINT  [new — EC3]")
    print(f"{'─'*50}")
    best_r2 = -np.inf; best_params = None
    for trial in range(n_iter):
        trial_params = {k: random.choice(v)
                        for k, v in HPO_SEARCH_SPACES['SAINT'].items()}
        val_r2 = run_hpo_trial_saint(
            trial_params, train_data, val_data,
            label_encoder, n_categories,
            seed=CV_RANDOM_STATE + trial)
        if val_r2 > best_r2:
            best_r2     = val_r2
            best_params = trial_params.copy()
            print(f"  Trial {trial+1:>2}/{n_iter}: "
                  f"R2={val_r2:.4f} ← new best {trial_params}")
        else:
            print(f"  Trial {trial+1:>2}/{n_iter}: R2={val_r2:.4f}")
    best_params_all['SAINT'] = best_params
    print(f"\n  Best params for SAINT: {best_params}")
    print(f"  Best val R2: {best_r2:.4f}")

    # ── Summary and save ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("HPO COMPLETE — OPTIMAL CONFIGURATIONS")
    print(f"{'='*70}")
    for model_name, params in best_params_all.items():
        print(f"\n{model_name}:")
        for k, v in params.items():
            print(f"  {k}: {v}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / 'hpo_optimal_params.json'
    with open(out_path, 'w') as f:
        json.dump(best_params_all, f, indent=2)
    print(f"\nSaved to {out_path}")

    return best_params_all
