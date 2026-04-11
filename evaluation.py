"""
evaluation.py
=============
Statistical significance testing, publication table generation,
subset analysis, and plotting.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import wilcoxon as wilcoxon_test

from config import RESULTS_DIR


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — SIGNIFICANCE TESTS
# ═════════════════════════════════════════════════════════════════════════════
def compute_significance_kfold(results_dict: dict) -> tuple:
    """
    For each method, test whether the best model's fold-level R² scores
    are significantly higher using:
      1. Paired t-test  (parametric)
      2. Wilcoxon signed-rank test  (non-parametric)  [REVISED: EC4]

    Both tests are reported for transparency.  With K=5 folds the minimum
    achievable two-tailed Wilcoxon p-value is 0.0625 (all diffs same sign).

    Returns
    -------
    best_r2_method      : str  — name of model with highest mean R²
    significance_markers: dict — {method: marker string e.g. '*', '**', ''}
    """
    print(f"\n{'='*60}")
    print("PHASE 1: STATISTICAL SIGNIFICANCE TEST (Based on R2)")
    print(f"{'='*60}")

    methods = list(results_dict.keys())
    if not methods:
        return None, {}

    best_r2_method = max(methods, key=lambda m: results_dict[m]['r2_mean'])
    print(f"\nBest Model (By R2 Mean): {best_r2_method}")
    print(f"   R2 = {results_dict[best_r2_method]['r2_mean']:.4f}"
          f" ± {results_dict[best_r2_method]['r2_std']:.4f}")

    print(f"\n{'Method':<25} {'Mean R2':<12} "
          f"{'t-test p':<12} {'Wilcoxon p':<14} {'Significance':<15}")
    print("-" * 78)

    significance_markers = {}
    best_scores = results_dict[best_r2_method]['r2_values']

    print(f"\nNOTE: With K=5 paired observations, the minimum achievable\n"
          f"two-tailed Wilcoxon p-value is 0.0625. Both tests are\n"
          f"reported for transparency.\n")

    for method in methods:
        if method == best_r2_method:
            print(f"{method:<25} "
                  f"{results_dict[method]['r2_mean']:.4f}"
                  f"       -            -              (reference)")
            significance_markers[method] = ""
            continue

        method_scores = results_dict[method]['r2_values']
        if len(best_scores) != len(method_scores):
            print(f"{method:<25} (score count mismatch — skipped)")
            continue

        # Paired t-test
        t_stat, p_ttest = stats.ttest_rel(best_scores, method_scores)

        try:
            w_stat, p_wilcoxon = wilcoxon_test(
                best_scores, method_scores, alternative='greater')
            w_str = f"{p_wilcoxon:.4f}"
        except Exception:
            # Wilcoxon fails when all differences are zero
            p_wilcoxon = 1.0
            w_str = "N/A"

        # Significance marker based on t-test 
        if   p_ttest < 0.001: marker = "***"
        elif p_ttest < 0.01:  marker = "**"
        elif p_ttest < 0.05:  marker = "*"
        else:                  marker = "ns"

        significance_markers[method] = marker if marker != "ns" else ""
        print(f"{method:<25} "
              f"{results_dict[method]['r2_mean']:.4f}       "
              f"{p_ttest:.4f}       "
              f"{w_str:<14} "
              f"{marker if marker != 'ns' else 'not significant'}")

    return best_r2_method, significance_markers


# ═════════════════════════════════════════════════════════════════════════════
def generate_phase1_table(results_dict: dict,
                           significance_markers: dict, K: int):
    print(f"\n{'='*80}")
    print(f"PHASE 1 (K={K}): PUBLICATION TABLE")
    print(f"{'='*80}\n")
    print(f"{'Method':<25} {'R2 (mean±std)':<22} "
          f"{'MAPE (%)':<18} {'MAE':<18} "
          f"{'Training (s)':<15} {'Memory (MB)':<15} {'Inference (ms)'}")
    print("-" * 120)

    # [REVISED: RENAME] 'ConfigurableEnsemble' → 'GatedFusion'
    method_order = ['Baseline', 'Dual-Stream', 'Adaptive', 'GatedFusion']
    for method in method_order:
        if method not in results_dict:
            continue
        r   = results_dict[method]
        sig = significance_markers.get(method, '')
        print(f"{method:<25} "
              f"{r['r2_mean']:.4f} ± {r['r2_std']:.4f}{sig:<3}  "
              f"{r['mape_mean']:.1f} ± {r['mape_std']:.1f}      "
              f"{r['mae_mean']:.2f} ± {r['mae_std']:.2f}   "
              f"{r['train_mean']:.2f} ± {r['train_std']:.2f}   "
              f"{r['peak_mean']:.2f} ± {r['peak_std']:.2f}   "
              f"{r['inference_mean']:.4f} ± {r['inference_std']:.4f}")

    print(f"\nSignificance (* p<0.05, ** p<0.01, *** p<0.001) "
          f"vs. best R2 model (Paired t-test + Wilcoxon, K={K})")
    print("Note: Wilcoxon p-values also reported in significance "
          "test section above.")  # [REVISED: EC4]


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2C — STATISTICAL TESTS
# Paired t-test + Wilcoxon on interaction effect (missing vs complete gap)
# ═════════════════════════════════════════════════════════════════════════════
def run_statistical_tests(ss_model: dict, ss_fusion: dict,
                           model_name: str):
    """
    Tests whether Gated Fusion's advantage is significantly greater on
    missing-engagement records than on complete records compared to a
    given baseline.

    ss_model / ss_fusion: subset_summary dicts with keys
        'complete_values', 'missing_values'
    """
    if ss_model is None or ss_fusion is None:
        print(f"\n  {model_name}: insufficient data for statistical test")
        return

    n = min(len(ss_fusion['complete_values']),
            len(ss_model['complete_values']))

    fusion_complete = np.array(ss_fusion['complete_values'][:n])
    fusion_missing  = np.array(ss_fusion['missing_values'][:n])
    model_complete  = np.array(ss_model['complete_values'][:n])
    model_missing   = np.array(ss_model['missing_values'][:n])

    gap_complete = fusion_complete - model_complete
    gap_missing  = fusion_missing  - model_missing
    interaction  = gap_missing - gap_complete

    print(f"\n  Gated Fusion vs {model_name}:")
    print(f"  {'Fold':<6} {'Gap(Complete)':<16} "
          f"{'Gap(Missing)':<16} {'Interaction'}")
    print(f"  {'─'*52}")
    for i in range(n):
        print(f"  {i+1:<6} {gap_complete[i]:+.4f}         "
              f"{gap_missing[i]:+.4f}         "
              f"{interaction[i]:+.4f}")

    mean_int = np.mean(interaction)
    print(f"\n  Mean interaction: {mean_int:+.4f}")
    print(f"  Direction (Fusion advantages missing more than complete): "
          f"{'YES' if mean_int > 0 else 'NO'}")

    # Paired t-test
    t_stat, p_ttest = stats.ttest_1samp(interaction, 0)
    sig_t = ('***' if p_ttest < 0.001 else '**' if p_ttest < 0.01
             else '*' if p_ttest < 0.05 else 'ns')
    print(f"\n  Paired t-test (interaction vs 0): "
          f"t={t_stat:.4f}, p={p_ttest:.4f} ({sig_t})")

    # Wilcoxon signed-rank test
    try:
        w_stat, p_wilcoxon = wilcoxon_test(
            interaction, alternative='greater')
        sig_w = ('***' if p_wilcoxon < 0.001 else
                 '**'  if p_wilcoxon < 0.01  else
                 '*'   if p_wilcoxon < 0.05  else 'ns')
        print(f"  Wilcoxon signed-rank (one-tailed, interaction > 0): "
              f"W={w_stat:.4f}, p={p_wilcoxon:.4f} ({sig_w})")
        print(f"  Note: minimum achievable p with n={n} is 0.0625")
    except Exception as e:
        print(f"  Wilcoxon: could not compute ({e})")

    # Win rates
    print(f"\n  Win rates (Fusion > {model_name}):")
    print(f"    Complete data: {np.sum(gap_complete > 0)}/{n} folds")
    print(f"    Missing data:  {np.sum(gap_missing  > 0)}/{n} folds")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — FULL ANALYSIS OUTPUT
# ═════════════════════════════════════════════════════════════════════════════
def analyze_phase2_results(results_dict: dict, subset_dict: dict, K: int):
    """
    Prints Phase 2A (general), 2B (subset), and 2C (statistical tests),
    then saves a JSON summary.

    results_dict keys: 'GatedFusion', 'XGBoost-Zero', 'XGBoost-Native', 'SAINT'
    subset_dict  keys: same — values have 'complete_values', 'missing_values'
    """
    # ── Phase 2A ─────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("PHASE 2A: GENERAL PERFORMANCE SUMMARY (K=5)")
    print(f"{'='*80}")
    print(f"\n{'Method':<22} {'R2 (mean±std)':<22} "
          f"{'MAE':<18} {'MAPE(%)':<12} {'Time(s)'}")
    print("─" * 82)
    for name, s in results_dict.items():
        print(f"{name:<22} "
              f"{s['r2_mean']:.4f} ± {s['r2_std']:.4f}   "
              f"{s['mae_mean']:.2f} ± {s['mae_std']:.2f}   "
              f"{s['mape_mean']:.1f} ± {s['mape_std']:.1f}   "
              f"{s['time_mean']:.2f} ± {s['time_std']:.2f}")

    # ── Phase 2B ─────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("PHASE 2B: SUBSET ANALYSIS (Complete vs Missing Data)")
    print(f"{'='*80}")
    print(f"\n{'Method':<22} {'Complete R2':<22} "
          f"{'Missing R2':<22} {'Diff (M-C)'}")
    print("─" * 72)
    for name, ss in subset_dict.items():
        if ss is None:
            continue
        diff = ss['missing_mean'] - ss['complete_mean']
        print(f"{name:<22} "
              f"{ss['complete_mean']:.4f} ± {ss['complete_std']:.4f}   "
              f"{ss['missing_mean']:.4f} ± {ss['missing_std']:.4f}   "
              f"{diff:+.4f}")

    # ── Phase 2C ─────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("PHASE 2C: STATISTICAL TESTS")
    print("  Comparing Gated Fusion vs each baseline")
    print(f"{'='*80}")
    print("\nNOTE: With K=5 paired observations (df=4), both tests have\n"
          "limited statistical power. Minimum achievable Wilcoxon\n"
          "two-tailed p with n=5 is 0.0625.\n")

    ss_fusion = subset_dict.get('GatedFusion')
    for name in ['XGBoost-Zero', 'XGBoost-Native', 'SAINT']:
        run_statistical_tests(subset_dict.get(name), ss_fusion, name)

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    save_data = {
        'general': {k: {sk: sv for sk, sv in v.items()
                        if sk not in ('r2_values',)}
                    for k, v in results_dict.items()},
        'subsets': {k: v for k, v in subset_dict.items() if v is not None},
    }
    out_path = RESULTS_DIR / 'phase2_extended_results.json'
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — ACTUAL VS PREDICTED
# ═════════════════════════════════════════════════════════════════════════════
def plot_actual_vs_predicted(all_predictions: dict, summary_stats: dict):
    """
    Creates Figure 5: 2×2 grid of actual vs predicted scatter plots.
    Each subplot shows one Phase-1 model.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # [REVISED: RENAME] 'ConfigurableEnsemble' → 'GatedFusion'
    method_order = ['Baseline', 'Dual-Stream', 'Adaptive', 'GatedFusion']
    titles = {
        'Baseline':    'Baseline\nR² = {r2:.3f}, MAPE = {mape:.1f}%',
        'Dual-Stream': 'Dual-Stream\nR² = {r2:.3f}, MAPE = {mape:.1f}%',
        'Adaptive':    'Adaptive\nR² = {r2:.3f}, MAPE = {mape:.1f}%',
        'GatedFusion': 'Gated Fusion\nR² = {r2:.3f}, MAPE = {mape:.1f}%',
    }
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for method, pos in zip(method_order, positions):
        if (method not in all_predictions or
                not all_predictions[method]['y_true']):
            continue
        ax = fig.add_subplot(gs[pos[0], pos[1]])

        y_true   = np.array(all_predictions[method]['y_true'])
        y_pred   = np.array(all_predictions[method]['y_pred'])
        scatter  = ax.scatter(y_true, y_pred, alpha=0.5, s=30,
                              c=y_true, cmap='viridis', edgecolors='none')
        max_val  = max(y_true.max(), y_pred.max())
        min_val  = min(y_true.min(), y_pred.min())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')

        r2   = summary_stats[method]['r2_mean']
        mape = summary_stats[method]['mape_mean']
        ax.set_title(titles[method].format(r2=r2, mape=mape),
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Actual Sales Quantity', fontsize=10)
        ax.set_ylabel('Predicted Sales Quantity', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        if pos[1] == 1:
            plt.colorbar(scatter, ax=ax).set_label('Actual Value', fontsize=9)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle('Actual vs. Predicted Sales Quantities Comparison',
                 fontsize=14, fontweight='bold', y=0.98)
    out = RESULTS_DIR / 'figure5_actual_vs_predicted.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure 5 saved to {out}")
