"""
run_phase2.py
=============
Phase 2: Extended Baseline Comparison (K=5).

Usage
-----
python run_phase2.py
  (reads optimal params from results/hpo_optimal_params.json)

Outputs
-------
results/phase2_extended_results.json
"""

import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import (K_FOLDS, CV_RANDOM_STATE, RESULTS_DIR,
                    N_SALES_FEATURES, N_VIEW_FEATURES, N_TOTAL_FEATURES,
                    OPTIMAL_PARAMS)
from data_processor import ImprovedDataProcessor, build_label_encoder
from models import GatedFusionTransformer
from training import (train_script2_model_run, train_xgboost_zero,
                       train_xgboost_native, train_saint)
from evaluation import analyze_phase2_results

RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SUBSET EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_subset(model_name, model_instance,
                    processor, train_data, subset_data,
                    label_encoder, n_categories,
                    sales_dim, input_dim, params, seed,
                    is_saint=False, is_xgb_zero=False, is_xgb_native=False):
    if is_xgb_zero:
        return train_xgboost_zero(
            train_data, subset_data, processor,
            seed, label_encoder, xgb_params=params)['metrics']['r2']
    elif is_xgb_native:
        return train_xgboost_native(
            train_data, subset_data, processor,
            seed, label_encoder, xgb_params=params)['metrics']['r2']
    elif is_saint:
        return train_saint(
            train_data, subset_data, processor, seed, label_encoder,
            d_model=params['d_model'], nhead=params['nhead'],
            n_layers=params['n_layers'], lr=params['lr'],
            weight_decay=params['weight_decay'], dropout=params['dropout'],
            epochs=params.get('epochs', 150),
            n_categories=n_categories)['metrics']['r2']
    else:
        # [REVISED: RENAME] GatedFusion
        return train_script2_model_run(
            model_instance, processor, train_data, subset_data,
            'GatedFusion', seed, label_encoder,
            n_categories, sales_dim, input_dim,
            lr=params['lr'], weight_decay=params['weight_decay'],
            n_runs=3)['metrics']['r2']


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def run_phase2_extended(K, full_data, label_encoder, model_dims,
                         n_categories, stratify_groups,
                         optimal_params, saint_params):

    print("\n" + "="*80)
    print(f"PHASE 2: EXTENDED BASELINE COMPARISON (K={K})")
    print("  Models: Gated Fusion | XGBoost-Zero | XGBoost-Native | SAINT")
    print("="*80)

    sales_dim, view_dim, input_dim = model_dims
    kfold = StratifiedKFold(
        n_splits=K, shuffle=True, random_state=CV_RANDOM_STATE)

    # [REVISED: RENAME] params key 'GatedFusion' replaces 'ConfigurableEnsemble'
    params_gf  = optimal_params['GatedFusion']
    params_xgb = optimal_params['XGBoost']

    results_fusion = []; results_xgb_z = []
    results_xgb_n  = []; results_saint = []
    subset_fusion  = []; subset_xgb_z  = []
    subset_xgb_n   = []; subset_saint  = []

    for fold, (train_idx, test_idx) in enumerate(
            kfold.split(full_data, stratify_groups), 1):

        print(f"\n{'─'*60}\nFOLD {fold}/{K}\n{'─'*60}")
        train_data = full_data.iloc[train_idx]
        test_data  = full_data.iloc[test_idx]

        has_view      = test_data['view_count'].fillna(0) > 0
        complete_data = test_data[has_view].copy()
        missing_data  = test_data[~has_view].copy()
        print(f"  Test: {len(test_data)} total | "
              f"{len(complete_data)} complete | {len(missing_data)} missing")

        do_subset = len(complete_data) > 0 and len(missing_data) > 0

        # ── 1. GATED FUSION ───────────────────────────────────────────────
        print("\n  [1/4] Gated Fusion...")
        proc_f = ImprovedDataProcessor()
        # [REVISED: RENAME] was ConfigurableEnsembleTransformer
        model_kwargs = {
            'sales_dim': sales_dim, 'full_dim': input_dim,
            'n_categories': n_categories, 'nhead': 4, 'num_layers': 1,
            'd_model': params_gf.get('d_model', 64),
            'dropout': params_gf.get('dropout', 0.1),
            'category_emb_dim': params_gf.get('category_emb_dim', 8)}
        model_f = GatedFusionTransformer(**model_kwargs)
        res_f = train_script2_model_run(
            model_f, proc_f, train_data, test_data,
            'GatedFusion', CV_RANDOM_STATE + fold,
            label_encoder, n_categories, sales_dim, input_dim,
            lr=params_gf['lr'], weight_decay=params_gf['weight_decay'],
            n_runs=3)
        results_fusion.append(res_f)
        print(f"     General R2: {res_f['metrics']['r2']:.4f}")

        if do_subset:
            proc_fc = ImprovedDataProcessor()
            r2_fc = evaluate_subset(
                'GatedFusion', GatedFusionTransformer(**model_kwargs),
                proc_fc, train_data, complete_data,
                label_encoder, n_categories, sales_dim, input_dim,
                params_gf, CV_RANDOM_STATE + fold)
            proc_fm = ImprovedDataProcessor()
            r2_fm = evaluate_subset(
                'GatedFusion', GatedFusionTransformer(**model_kwargs),
                proc_fm, train_data, missing_data,
                label_encoder, n_categories, sales_dim, input_dim,
                params_gf, CV_RANDOM_STATE + fold)
            subset_fusion.append({'complete': r2_fc, 'missing': r2_fm})
            print(f"     Complete: {r2_fc:.4f} | Missing: {r2_fm:.4f}")
        else:
            subset_fusion.append(None)

        # ── 2. XGBOOST-ZERO ───────────────────────────────────────────────
        print("\n  [2/4] XGBoost-Zero...")
        proc_xz = ImprovedDataProcessor()
        res_xz  = train_xgboost_zero(train_data, test_data, proc_xz,
                                       CV_RANDOM_STATE + fold,
                                       label_encoder, xgb_params=params_xgb)
        results_xgb_z.append(res_xz)
        print(f"     General R2: {res_xz['metrics']['r2']:.4f}")

        if do_subset:
            proc_xzc = ImprovedDataProcessor()
            r2_xzc = evaluate_subset(
                'XGBoost-Zero', None, proc_xzc, train_data, complete_data,
                label_encoder, n_categories, sales_dim, input_dim,
                params_xgb, CV_RANDOM_STATE + fold, is_xgb_zero=True)
            proc_xzm = ImprovedDataProcessor()
            r2_xzm = evaluate_subset(
                'XGBoost-Zero', None, proc_xzm, train_data, missing_data,
                label_encoder, n_categories, sales_dim, input_dim,
                params_xgb, CV_RANDOM_STATE + fold, is_xgb_zero=True)
            subset_xgb_z.append({'complete': r2_xzc, 'missing': r2_xzm})
            print(f"     Complete: {r2_xzc:.4f} | Missing: {r2_xzm:.4f}")
        else:
            subset_xgb_z.append(None)

        # ── 3. XGBOOST-NATIVE  ────────────────────────────
        print("\n  [3/4] XGBoost-Native...")
        proc_xn = ImprovedDataProcessor()
        res_xn  = train_xgboost_native(train_data, test_data, proc_xn,
                                        CV_RANDOM_STATE + fold,
                                        label_encoder, xgb_params=params_xgb)
        results_xgb_n.append(res_xn)
        print(f"     General R2: {res_xn['metrics']['r2']:.4f}")

        if do_subset:
            proc_xnc = ImprovedDataProcessor()
            r2_xnc = evaluate_subset(
                'XGBoost-Native', None, proc_xnc, train_data, complete_data,
                label_encoder, n_categories, sales_dim, input_dim,
                params_xgb, CV_RANDOM_STATE + fold, is_xgb_native=True)
            proc_xnm = ImprovedDataProcessor()
            r2_xnm = evaluate_subset(
                'XGBoost-Native', None, proc_xnm, train_data, missing_data,
                label_encoder, n_categories, sales_dim, input_dim,
                params_xgb, CV_RANDOM_STATE + fold, is_xgb_native=True)
            subset_xgb_n.append({'complete': r2_xnc, 'missing': r2_xnm})
            print(f"     Complete: {r2_xnc:.4f} | Missing: {r2_xnm:.4f}")
        else:
            subset_xgb_n.append(None)

        # ── 4. SAINT  ───────────────────────────────────────
        print("\n  [4/4] SAINT...")
        proc_s = ImprovedDataProcessor()
        res_s  = train_saint(
            train_data, test_data, proc_s,
            CV_RANDOM_STATE + fold, label_encoder,
            d_model=saint_params['d_model'],
            nhead=saint_params['nhead'],
            n_layers=saint_params['n_layers'],
            lr=saint_params['lr'],
            weight_decay=saint_params['weight_decay'],
            dropout=saint_params['dropout'],
            epochs=saint_params.get('epochs', 150),
            n_categories=n_categories)
        results_saint.append(res_s)
        print(f"     General R2: {res_s['metrics']['r2']:.4f}")

        if do_subset:
            proc_sc = ImprovedDataProcessor()
            r2_sc = evaluate_subset(
                'SAINT', None, proc_sc, train_data, complete_data,
                label_encoder, n_categories, sales_dim, input_dim,
                saint_params, CV_RANDOM_STATE + fold, is_saint=True)
            proc_sm = ImprovedDataProcessor()
            r2_sm = evaluate_subset(
                'SAINT', None, proc_sm, train_data, missing_data,
                label_encoder, n_categories, sales_dim, input_dim,
                saint_params, CV_RANDOM_STATE + fold, is_saint=True)
            subset_saint.append({'complete': r2_sc, 'missing': r2_sm})
            print(f"     Complete: {r2_sc:.4f} | Missing: {r2_sm:.4f}")
        else:
            subset_saint.append(None)

    # ── Build summary dicts and call evaluation ────────────────────────────
    def summarize(results, name):
        r2s   = [r['metrics']['r2']   for r in results]
        maes  = [r['metrics']['mae']  for r in results]
        mapes = [r['metrics']['mape'] for r in results]
        times = [r['train_time_sec']  for r in results]
        return {
            'name':      name,
            'r2_mean':   np.mean(r2s),   'r2_std':   np.std(r2s, ddof=1),
            'r2_values': r2s,
            'mae_mean':  np.mean(maes),  'mae_std':  np.std(maes, ddof=1),
            'mape_mean': np.mean(mapes), 'mape_std': np.std(mapes, ddof=1),
            'time_mean': np.mean(times), 'time_std': np.std(times, ddof=1),
        }

    def subset_summary(subset_list, name):
        valid = [s for s in subset_list if s is not None]
        if not valid:
            return None
        comp = [s['complete'] for s in valid]
        miss = [s['missing']  for s in valid]
        return {
            'name':            name,
            'complete_mean':   np.mean(comp), 'complete_std': np.std(comp, ddof=1),
            'complete_values': comp,
            'missing_mean':    np.mean(miss), 'missing_std':  np.std(miss, ddof=1),
            'missing_values':  miss,
        }

    results_dict = {
        'GatedFusion':    summarize(results_fusion,  'Gated Fusion'),
        'XGBoost-Zero':   summarize(results_xgb_z,   'XGBoost-Zero'),
        'XGBoost-Native': summarize(results_xgb_n,   'XGBoost-Native'),
        'SAINT':          summarize(results_saint,   'SAINT'),
    }
    subset_dict  = {
        'GatedFusion':    subset_summary(subset_fusion,  'Gated Fusion'),
        'XGBoost-Zero':   subset_summary(subset_xgb_z,   'XGBoost-Zero'),
        'XGBoost-Native': subset_summary(subset_xgb_n,   'XGBoost-Native'),
        'SAINT':          subset_summary(subset_saint,   'SAINT'),
    }

    analyze_phase2_results(results_dict, subset_dict, K)
    return results_dict, subset_dict


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    proc      = ImprovedDataProcessor()
    full_data = proc.load_full_dataset()
    le        = build_label_encoder(full_data)
    n_cats    = len(le.classes_)
    stratify_groups = proc.get_stratify_groups(
        full_data, test_size_for_rare_calc=1/K_FOLDS)

    model_dims = (N_SALES_FEATURES, N_VIEW_FEATURES, N_TOTAL_FEATURES)
    print(f"Dataset: {full_data.shape} | Categories: {n_cats}")

    # Load HPO results
    hpo_path = RESULTS_DIR / 'hpo_optimal_params.json'
    if hpo_path.exists():
        with open(hpo_path) as f:
            optimal_params = json.load(f)
        print(f"Loaded HPO params from {hpo_path}")
    else:
        print("HPO results not found — using config.py defaults.")
        optimal_params = OPTIMAL_PARAMS

    # SAINT params — use HPO result if available, else default
    saint_params = optimal_params.get('SAINT', {
        'd_model': 32, 'nhead': 2, 'n_layers': 2,
        'lr': 0.001, 'weight_decay': 1e-4,
        'dropout': 0.1, 'epochs': 150})

    run_phase2_extended(K_FOLDS, full_data, le, model_dims, n_cats,
                         stratify_groups, optimal_params, saint_params)


if __name__ == "__main__":
    main()
