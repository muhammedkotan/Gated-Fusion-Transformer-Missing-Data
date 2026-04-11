"""
data_processor.py
=================
Data loading, preprocessing, and feature engineering.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import (DATA_PATH, SHEET_NAME, CATEGORY_COL, TARGET_COL,
                    N_SALES_FEATURES, N_VIEW_FEATURES)


# ═════════════════════════════════════════════════════════════════════════════
class ImprovedDataProcessor:
    """
    Stateful preprocessor: fit scalers on train split, transform test split.
    """

    def __init__(self):
        self.scaler           = StandardScaler()   # all-feature scaler
        self.sales_scaler     = StandardScaler()   # sales-only scaler
        self.view_scaler      = StandardScaler()   # view-only scaler
        self.sales_scaler_xgb = StandardScaler()   # XGBoost sales scaler
        self.y_log_max        = None               # set during fit

    # ─────────────────────────────────────────────────────────────────────────
    # I/O
    # ─────────────────────────────────────────────────────────────────────────
    def load_full_dataset(self, excel_path: str = DATA_PATH) -> pd.DataFrame:
        return pd.read_excel(excel_path, sheet_name=SHEET_NAME)

    # ─────────────────────────────────────────────────────────────────────────
    # STRATIFICATION HELPER
    # ─────────────────────────────────────────────────────────────────────────
    def get_stratify_groups(self, data: pd.DataFrame,
                            test_size_for_rare_calc: float = 0.1
                            ) -> pd.Series:
        """
        Merge categories with too few samples into 'RARE_COMBINED' so that
        StratifiedKFold can place at least one sample in every fold.
        """
        category_counts    = data[CATEGORY_COL].value_counts()
        min_samples_needed = int(np.ceil(1 / test_size_for_rare_calc)) + 1
        rare_categories    = category_counts[
            category_counts < min_samples_needed].index.tolist()
        stratify_groups    = data[CATEGORY_COL].copy()
        if len(rare_categories) > 0:
            stratify_groups = stratify_groups.replace(
                rare_categories, 'RARE_COMBINED')
        return stratify_groups

    # ─────────────────────────────────────────────────────────────────────────
    # PIPELINE 1: 
    # ─────────────────────────────────────────────────────────────────────────
    def prepare_features_fair(self, data_df: pd.DataFrame,
                               label_encoder: LabelEncoder,
                               is_train: bool = True):
        """
        Build feature matrices used by all Transformer models and
        XGBoost-Zero. Missing engagement values are filled with zero.

        Returns
        -------
        X_sales       : ndarray (n, 5) — sales-only features
        X_view        : ndarray (n, 5) — engagement features (zero-filled)
        X_all         : ndarray (n,10) — concatenation of above
        y_norm        : ndarray (n,)   — log-normalised target [0, 1]
        y_log         : ndarray (n,)   — log1p(target)
        category_enc  : ndarray (n,)   — integer category labels
        view_mask     : ndarray (n,)   — bool True where engagement exists
        """
        # Category encoding
        categories = (data_df[CATEGORY_COL]
                      .fillna('Unknown').astype(str).str.strip())
        category_encoded = np.array([
            label_encoder.transform([cat])[0]
            if cat in label_encoder.classes_ else
            (label_encoder.transform(['Unknown'])[0]
             if 'Unknown' in label_encoder.classes_ else 0)
            for cat in categories
        ])
        target = data_df[TARGET_COL].fillna(0).values

        # Sales features (always complete)
        net_sales_amount = data_df['net_sales_amount'].fillna(0).values
        sales_features   = [net_sales_amount, np.log1p(net_sales_amount)]
        for col in ('gross_sales_amount', 'cost', 'discount_amount'):
            if col in data_df.columns:
                sales_features.append(data_df[col].fillna(0).values)

        # Engagement features — zero-fill where structurally absent
        view_count    = data_df['view_count'].fillna(0).values
        view_features = [view_count, np.log1p(view_count)]
        for col in ('view_duration', 'avg_view_duration'):
            if col in data_df.columns:
                view_features.append(data_df[col].fillna(0).values)
        view_features.append((view_count > 0).astype(int))   # binary flag

        X_sales   = np.column_stack(sales_features)
        X_view    = np.column_stack(view_features)
        X_all     = np.column_stack([X_sales, X_view])
        view_mask = (view_count > 0)

        # Target normalisation
        y     = np.maximum(target, 0.1)
        y_log = np.log1p(y)
        if is_train:
            self.y_log_max = y_log.max()
            if self.y_log_max == 0 or self.y_log_max is None:
                self.y_log_max = 1.0
        elif self.y_log_max is None:
            self.y_log_max = 1.0
        y_norm = y_log / self.y_log_max

        return X_sales, X_view, X_all, y_norm, y_log, category_encoded, view_mask

    # ─────────────────────────────────────────────────────────────────────────
    # PIPELINE 2: NaN-PRESERVING  (XGBoost-Native)
    # [REVISED: EC3] New method — preserves structural missingness so that
    # XGBoost's sparsity-aware algorithm can learn optimal default paths
    # for missing engagement values rather than treating zeros as signal.
    # ─────────────────────────────────────────────────────────────────────────
    def prepare_features_xgb_native(self, data_df: pd.DataFrame,
                                     label_encoder: LabelEncoder,
                                     is_train: bool = True):
        """
        Build feature matrix for XGBoost-Native.
        view_duration and avg_view_duration are set to NaN where the
        record has no engagement data (view_count == 0).

        Returns
        -------
        X_all_native  : ndarray (n,10) — engagement cols may contain NaN
        y_log         : ndarray (n,)   — log1p(target)
        category_enc  : ndarray (n,)   — integer category labels
        view_mask     : ndarray (n,)   — bool
        """
        # Category encoding (identical to fair pipeline)
        categories = (data_df[CATEGORY_COL]
                      .fillna('Unknown').astype(str).str.strip())
        category_encoded = np.array([
            label_encoder.transform([cat])[0]
            if cat in label_encoder.classes_ else
            (label_encoder.transform(['Unknown'])[0]
             if 'Unknown' in label_encoder.classes_ else 0)
            for cat in categories
        ])
        target = data_df[TARGET_COL].fillna(0).values

        # Sales features — identical to fair pipeline
        net_sales_amount = data_df['net_sales_amount'].fillna(0).values
        sales_features   = [net_sales_amount, np.log1p(net_sales_amount)]
        for col in ('gross_sales_amount', 'cost', 'discount_amount'):
            if col in data_df.columns:
                sales_features.append(data_df[col].fillna(0).values)
        X_sales = np.column_stack(sales_features)

        # Engagement features — NaN preserved where structurally absent
        view_count_raw = data_df['view_count'].values
        view_mask      = (view_count_raw > 0)
        view_count_log = np.log1p(view_count_raw)

        # view_duration: retain genuine value where present; NaN where absent
        if 'view_duration' in data_df.columns:
            view_duration = np.where(
                view_mask, data_df['view_duration'].values, np.nan)
        else:
            view_duration = np.where(view_mask, 0.0, np.nan)

        # avg_view_duration: NaN where structurally absent
        if 'avg_view_duration' in data_df.columns:
            avg_view_dur = np.where(
                view_mask,
                data_df['avg_view_duration'].fillna(0).values,
                np.nan)
        else:
            avg_view_dur = np.where(view_mask, 0.0, np.nan)

        # Binary indicator: always 0 or 1, no NaN
        view_indicator = view_mask.astype(float)

        X_view_native = np.column_stack([
            view_count_raw,
            view_count_log,
            view_duration,
            avg_view_dur,
            view_indicator,
        ])
        X_all_native = np.column_stack([X_sales, X_view_native])

        # Target (identical to fair pipeline)
        y     = np.maximum(target, 0.1)
        y_log = np.log1p(y)
        if is_train:
            self.y_log_max = y_log.max()
            if self.y_log_max == 0 or self.y_log_max is None:
                self.y_log_max = 1.0
        elif self.y_log_max is None:
            self.y_log_max = 1.0

        return X_all_native, y_log, category_encoded, view_mask


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_label_encoder(full_data: pd.DataFrame) -> LabelEncoder:
    """Fit a LabelEncoder on all categories in the full dataset."""
    all_categories = (full_data[CATEGORY_COL]
                      .fillna('Unknown').astype(str).str.strip())
    le = LabelEncoder()
    le.fit(all_categories)
    return le
