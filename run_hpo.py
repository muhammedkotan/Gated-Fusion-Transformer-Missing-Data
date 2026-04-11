"""
run_hpo.py
==========
Standalone hyperparameter optimisation runner.

Run this BEFORE run_phase1.py and run_phase2.py to find optimal
hyperparameters for all models.

Usage
-----
python run_hpo.py

Outputs
-------
results/hpo_optimal_params.json   — best params for each model
"""

from config import (K_FOLDS, N_SALES_FEATURES, N_VIEW_FEATURES,
                    N_TOTAL_FEATURES, N_ITER_HPO)
from data_processor import ImprovedDataProcessor, build_label_encoder
from hpo import run_hpo


def main():
    print("Loading data for HPO...")
    proc      = ImprovedDataProcessor()
    full_data = proc.load_full_dataset()
    le        = build_label_encoder(full_data)
    n_cats    = len(le.classes_)
    stratify_groups = proc.get_stratify_groups(
        full_data, test_size_for_rare_calc=1/K_FOLDS)

    sales_dim = N_SALES_FEATURES
    view_dim  = N_VIEW_FEATURES
    input_dim = N_TOTAL_FEATURES
    print(f"Dataset: {full_data.shape} | Categories: {n_cats}")

    optimal_params = run_hpo(
        full_data, le, n_cats, sales_dim, input_dim,
        stratify_groups, n_iter=N_ITER_HPO)

    print("\nHPO complete.")
    print("Next steps:")
    print("  1. Check results/hpo_optimal_params.json")
    print("  2. Run: python run_phase1.py")
    print("  3. Run: python run_phase2.py")


if __name__ == "__main__":
    main()
