"""
run_phase1.py
=============
Phase 1: K-Fold Transformer Tournament.

Runs the 4-model ablation study (Baseline, Dual-Stream, Adaptive,
GatedFusion) with optimal HPO parameters

Usage
-----
python run_phase1.py
  (reads optimal params from results/hpo_optimal_params.json)

Outputs
-------
results/phase1_results.json   
"""

import copy
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from config import (K_FOLDS, CV_RANDOM_STATE, RESULTS_DIR,
                    N_SALES_FEATURES, N_VIEW_FEATURES, N_TOTAL_FEATURES,
                    OPTIMAL_PARAMS)
from data_processor import ImprovedDataProcessor, build_label_encoder
from models import (BaselineTransformer, DualStreamTransformer,
                    AdaptiveWeightingTransformer, GatedFusionTransformer)
from training import (train_script1_model_run, train_script2_model_run)
from evaluation import (compute_significance_kfold,
                         generate_phase1_table,
                         plot_actual_vs_predicted)

RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 TOURNAMENT
# ─────────────────────────────────────────────────────────────────────────────
def run_phase1_tournament(K, full_data, label_encoder,
                           model_definitions, model_dims,
                           n_categories, stratify_groups,
                           optimal_params):
    print("\n" + "="*80)
    print(f"PHASE 1: TRANSFORMER TOURNAMENT (K={K}) STARTING")
    print("   Using optimized parameters...")
    print("="*80)

    sales_dim, view_dim, input_dim = model_dims
    kfold = StratifiedKFold(
        n_splits=K, shuffle=True, random_state=CV_RANDOM_STATE)

    fold_results_agg = {
        name: [] for name in model_definitions}
    all_predictions = {
        name: {'y_true': [], 'y_pred': []}
        for name in fold_results_agg}

    for fold, (train_idx, test_idx) in enumerate(
            kfold.split(full_data, stratify_groups), 1):

        print(f"\n--- Fold {fold}/{K} ---")
        train_data = full_data.iloc[train_idx]
        test_data  = full_data.iloc[test_idx]

        for model_name in fold_results_agg:
            print(f"   Training model: {model_name}")
            params       = optimal_params[model_name]
            model_kwargs = copy.deepcopy(
                model_definitions[model_name]['base_kwargs'])
            model_kwargs['d_model'] = params.get('d_model', 32)
            model_kwargs['dropout'] = params.get('dropout', 0.1)

            if model_name == 'Adaptive' and 'alpha' in params:
                model_kwargs['alpha'] = params['alpha']
            # [REVISED: RENAME] was 'ConfigurableEnsemble'
            if model_name == 'GatedFusion' and 'category_emb_dim' in params:
                model_kwargs['category_emb_dim'] = params['category_emb_dim']

            model_instance = model_definitions[model_name]['class'](
                **model_kwargs)
            processor = ImprovedDataProcessor()

            try:
                # [REVISED: RENAME] was 'ConfigurableEnsemble'
                if model_name == 'GatedFusion':
                    run_metrics = model_definitions[model_name]['train_fn'](
                        model_instance, processor,
                        train_data, test_data, model_name,
                        CV_RANDOM_STATE + fold,
                        label_encoder, n_categories, sales_dim, input_dim,
                        lr=params['lr'],
                        weight_decay=params['weight_decay'],
                        verbose=True)
                else:
                    run_metrics = model_definitions[model_name]['train_fn'](
                        model_instance, processor,
                        train_data, test_data, model_name,
                        CV_RANDOM_STATE + fold,
                        label_encoder,
                        lr=params['lr'],
                        weight_decay=params['weight_decay'])

                fold_results_agg[model_name].append(run_metrics)
                print(f"       Result: "
                      f"R2={run_metrics['metrics']['r2']:.4f} "
                      f"(Time: {run_metrics['train_time_sec']:.1f}s)")

            except Exception as e:
                print(f"       ERROR: {model_name} (Fold {fold}): {e}")
                fold_results_agg[model_name].append(None)

    # ── Summary statistics ─────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"PHASE 1 (K={K}) SUMMARY STATISTICS")
    print("="*80)

    summary_stats = {}
    for model_name, fold_runs in fold_results_agg.items():
        valid_runs = [
            r for r in fold_runs
            if r is not None and r['metrics']['r2'] > -np.inf]
        if not valid_runs:
            print(f"\nModel: {model_name} (NO SUCCESSFUL FOLDS)")
            continue

        summary = {'method': model_name}
        for key in valid_runs[0]['metrics']:
            if key == 'val_loss':
                continue
            summary[f'{key}_values'] = [r['metrics'][key] for r in valid_runs]
        summary['train_values']     = [r['train_time_sec'] for r in valid_runs]
        summary['peak_values']      = [r['peak_memory_mb'] for r in valid_runs]
        summary['inference_values'] = [
            r['inference_time_ms_per_sample'] * 1000 for r in valid_runs]

        for key in list(summary.keys()):
            if key.endswith('_values'):
                metric_name = key.replace('_values', '')
                values      = summary[key]
                summary[f'{metric_name}_mean'] = np.mean(values)
                summary[f'{metric_name}_std']  = np.std(values, ddof=1)

        summary_stats[model_name] = summary
        print(f"\nModel: {model_name} ({len(valid_runs)}/{K} folds)")
        print(f"   R2   : {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
        print(f"   MAE  : {summary['mae_mean']:.2f} ± {summary['mae_std']:.2f}")
        print(f"   MAPE : {summary['mape_mean']:.1f} ± {summary['mape_std']:.1f}%")

    # ── Statistical tests [REVISED: EC4] Wilcoxon now included ───────────
    best_transformer, sig_markers = compute_significance_kfold(summary_stats)
    generate_phase1_table(summary_stats, sig_markers, K)
    plot_actual_vs_predicted(all_predictions, summary_stats)

    return best_transformer, summary_stats


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

    sales_dim  = N_SALES_FEATURES
    view_dim   = N_VIEW_FEATURES
    input_dim  = N_TOTAL_FEATURES
    model_dims = (sales_dim, view_dim, input_dim)

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

    # [REVISED: RENAME] key 'GatedFusion' replaces 'ConfigurableEnsemble'
    model_definitions = {
        'Baseline': {
            'class':      BaselineTransformer,
            'base_kwargs':{'input_dim': input_dim, 'n_categories': n_cats,
                           'nhead': 4},
            'train_fn':   train_script1_model_run,
        },
        'Dual-Stream': {
            'class':      DualStreamTransformer,
            'base_kwargs':{'sales_dim': sales_dim, 'view_dim': view_dim,
                           'n_categories': n_cats, 'nhead': 4},
            'train_fn':   train_script1_model_run,
        },
        'Adaptive': {
            'class':      AdaptiveWeightingTransformer,
            'base_kwargs':{'input_dim': input_dim, 'n_categories': n_cats,
                           'nhead': 4},
            'train_fn':   train_script1_model_run,
        },
        # [REVISED: RENAME] was 'ConfigurableEnsemble'
        'GatedFusion': {
            'class':      GatedFusionTransformer,
            'base_kwargs':{'sales_dim': sales_dim, 'full_dim': input_dim,
                           'n_categories': n_cats, 'nhead': 4, 'num_layers': 1},
            'train_fn':   train_script2_model_run,
        },
    }

    best_transformer, phase1_results = run_phase1_tournament(
        K_FOLDS, full_data, le, model_definitions, model_dims,
        n_cats, stratify_groups, optimal_params)

    print(f"\nPhase 1 complete. Best transformer: {best_transformer}")

    # Save results
    save_data = {
        method: {k: v for k, v in r.items() if k != 'method'}
        for method, r in phase1_results.items()}
    out = RESULTS_DIR / 'phase1_results.json'
    with open(out, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved to {out}")

    return best_transformer, phase1_results


if __name__ == "__main__":
    main()
