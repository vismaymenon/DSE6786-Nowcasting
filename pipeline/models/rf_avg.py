"""
rf_aggre.py
===========
Random Forest nowcasting model using simple quarterly aggregation of monthly data.

Pipeline
--------
  Step 1 — Load filled data
            Fetch ragged-edge-imputed data from Supabase (filled_md, filled_qd).
            Run test.py beforehand to populate these tables from the latest
            FRED release.

  Step 2 — Simple aggregation feature engineering
            Average the 3 monthly observations within each quarter per variable.
            115 monthly (averaged) + 208 quarterly = 323 features total.

  Step 3 — Random Forest with expanding-window CV
            TimeSeriesSplit (6 folds) tunes max_features only.
            max_depth=None and min_samples_leaf=1 (sklearn defaults).

  Step 4 — Evaluation and feature importance

Data split (264 quarterly GDP observations):
  Train     : obs   0–99   (100 quarters)  ← initial CV window
  Validation: obs 100–165  ( 66 quarters)  ← CV expands through here
  Test      : obs 166–263  (100 quarters)  ← strictly held out
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# ── RF config ─────────────────────────────────────────────────────────────────
N_TRAIN           = 100
N_TRAINVAL        = 166
N_SPLITS_CV       = 6
VAL_FOLD_SIZE     = 11
N_TREES_CV        = 300
N_TREES_FINAL     = 1000
MAX_FEATURES_GRID = ["sqrt", "log2", 0.1, 0.3]
TOP_N_FEATURES    = 20
RANDOM_STATE      = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
PROJECT_DIR  = PIPELINE_DIR.parent
DATA_DIR     = PROJECT_DIR / "data"

sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PROJECT_DIR))
from ragged_edge import read_table
from database.client import get_backend_client


# =============================================================================
# STEP 1 — Load ragged-edge-filled data from Supabase
# =============================================================================

def load_filled_data():
    """
    Fetch ragged-edge-imputed data from Supabase tables filled_md and filled_qd.

    Returns
    -------
    df_md_filled : pd.DataFrame — monthly FRED-MD, ragged edge imputed
    df_qd_filled : pd.DataFrame — quarterly FRED-QD-X, ragged edge imputed
    """
    print("Step 1 — Loading filled data from Supabase …")
    supabase = get_backend_client()

    df_md_filled = read_table(supabase, "filled_md")
    df_md_filled["sasdate"] = pd.to_datetime(df_md_filled["sasdate"])
    df_md_filled = df_md_filled.sort_values("sasdate").reset_index(drop=True)
    print(f"  filled_md : {df_md_filled.shape}")

    df_qd_filled = read_table(supabase, "filled_qd")
    df_qd_filled["sasdate"] = pd.to_datetime(df_qd_filled["sasdate"])
    df_qd_filled = df_qd_filled.sort_values("sasdate").reset_index(drop=True)
    print(f"  filled_qd : {df_qd_filled.shape}\n")

    return df_md_filled, df_qd_filled


# =============================================================================
# STEP 2 — SIMPLE AGGREGATION FEATURE ENGINEERING
# =============================================================================

def build_aggregated_features(df_md_filled: pd.DataFrame) -> pd.DataFrame:
    """
    Convert filled monthly data to quarterly by averaging the 3 months in each quarter.

    Quarter label convention (matches gdp.csv): first day of the last month in
    the quarter — e.g. Q1 1959 → 1959-03-01, Q2 1959 → 1959-06-01.

    COVID flag columns are excluded (kept as quarterly features instead).

    Returns DataFrame indexed by quarter label, shape (n_quarters, n_monthly_vars).
    """
    df = df_md_filled.copy().sort_values("sasdate").reset_index(drop=True)

    feature_cols = [c for c in df.columns
                    if c not in ("sasdate", "covid_crash", "covid_recover")]

    df["qtr_label"] = (
        df["sasdate"].dt.to_period("Q").dt.start_time
        + pd.DateOffset(months=2)
    )

    df_agg = (
        df.groupby("qtr_label")[feature_cols]
        .mean()
        .rename_axis("sasdate")
    )

    return df_agg


def build_feature_matrix(df_md_filled: pd.DataFrame, df_qd_filled: pd.DataFrame) -> tuple:
    """
    Assemble the full quarterly feature matrix X and GDP target y.

    Joins averaged monthly features and quarterly FRED-QD-X features,
    aligned to the GDP quarter dates.

    Returns
    -------
    X : pd.DataFrame  shape (n_quarters, n_features)
    y : pd.Series     shape (n_quarters,)
    """
    gdp = pd.read_csv(DATA_DIR / "gdp.csv", parse_dates=["sasdate"])
    gdp = gdp.set_index("sasdate").sort_index()

    df_agg = build_aggregated_features(df_md_filled)

    df_qd = df_qd_filled.copy()
    df_qd["sasdate"] = pd.to_datetime(df_qd["sasdate"])
    df_qd = df_qd.set_index("sasdate").sort_index()

    df_agg = df_agg.add_suffix("_md")
    X = df_agg.join(df_qd, how="inner").reindex(gdp.index)
    y = gdp["GDPC1_t"]

    valid = X.notna().all(axis=1) & y.notna()
    if (~valid).sum() > 0:
        print(f"  Warning: dropping {(~valid).sum()} rows with remaining NaNs.")
        X, y = X[valid], y[valid]

    print(f"Feature matrix: {X.shape[0]} quarters × {X.shape[1]} features")
    print(f"  Monthly (averaged) : {df_agg.shape[1]} features")
    print(f"  Quarterly (QD-X)   : {df_qd.shape[1]} features\n")
    
    return X, y


# =============================================================================
# STEP 3 — SPLIT + CROSS-VALIDATION + FITTING
# =============================================================================

def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Strict chronological split: train (100) / validation (66) / test (100)."""
    X_trainval = X.iloc[:N_TRAINVAL]
    X_test     = X.iloc[N_TRAINVAL:]
    y_trainval = y.iloc[:N_TRAINVAL]
    y_test     = y.iloc[N_TRAINVAL:]

    print("Data split:")
    print(f"  Train     : obs   0–{N_TRAIN-1:3d}  "
          f"({X.index[0].date()} → {X.index[N_TRAIN-1].date()})")
    print(f"  Validation: obs {N_TRAIN:3d}–{N_TRAINVAL-1:3d}  "
          f"({X.index[N_TRAIN].date()} → {X.index[N_TRAINVAL-1].date()})")
    print(f"  Test      : obs {N_TRAINVAL:3d}–{len(X)-1:3d}  "
          f"({X_test.index[0].date()} → {X_test.index[-1].date()})\n")

    return X_trainval, X_test, y_trainval, y_test


def tune_max_features(X_trainval: pd.DataFrame, y_trainval: pd.Series):
    """Select best max_features using 6-fold expanding-window TimeSeriesSplit CV."""
    tscv  = TimeSeriesSplit(n_splits=N_SPLITS_CV, test_size=VAL_FOLD_SIZE)
    X_arr = X_trainval.values
    y_arr = y_trainval.values

    print(f"Tuning max_features via {N_SPLITS_CV}-fold expanding-window CV "
          f"(n_estimators={N_TREES_CV}) …\n")

    cv_results = {}
    for mf in MAX_FEATURES_GRID:
        fold_rmses = []
        for train_idx, val_idx in tscv.split(X_arr):
            rf = RandomForestRegressor(
                n_estimators=N_TREES_CV,
                max_features=mf,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            rf.fit(X_arr[train_idx], y_arr[train_idx])
            pred = rf.predict(X_arr[val_idx])
            fold_rmses.append(np.sqrt(mean_squared_error(y_arr[val_idx], pred)))

        mean_rmse = np.mean(fold_rmses)
        cv_results[str(mf)] = mean_rmse
        fold_str = "  ".join(f"{r:.3f}" for r in fold_rmses)
        print(f"  max_features={str(mf):<6}  CV RMSE={mean_rmse:.4f}  [folds: {fold_str}]")

    best_key = min(cv_results, key=cv_results.get)
    print(f"\n>>> Best max_features = {best_key}  "
          f"(mean CV RMSE = {cv_results[best_key]:.4f})\n")

    try:
        return float(best_key)
    except ValueError:
        return best_key


def fit_final_model(X_trainval: pd.DataFrame, y_trainval: pd.Series, best_mf) -> RandomForestRegressor:
    """Refit on the full train+val set (166 obs) with the CV-selected max_features."""
    print(f"Fitting final model on train+val "
          f"(n_estimators={N_TREES_FINAL}, max_features={best_mf}) …")

    rf_final = RandomForestRegressor(
        n_estimators=N_TREES_FINAL,
        max_features=best_mf,
        max_depth=None,
        min_samples_leaf=1,
        max_samples=0.8,
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf_final.fit(X_trainval.values, y_trainval.values)
    return rf_final


# =============================================================================
# STEP 4 — POOS-COMPATIBLE MODEL WRAPPER
# =============================================================================

def rf_aggre_nowcast(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    POOS-compatible wrapper for the RF aggregation model.

    Follows the contract expected by poos.poos_validation():
      - Last row of X / last element of y is the test observation.
      - All preceding rows are used for training.

    Uses fixed max_features="sqrt" and N_TREES_CV trees for POOS speed.
    """
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_test  = X.iloc[[-1]]
    y_test_actual = float(y.iloc[-1])

    rf = RandomForestRegressor(
        n_estimators=N_TREES_CV,
        max_features="sqrt",
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train.values, y_train.values)

    y_train_predicted = rf.predict(X_train.values)
    y_test_predicted  = float(rf.predict(X_test.values)[0])

    return {
        "X_train":           X_train,
        "y_train":           y_train.values,
        "y_train_predicted": y_train_predicted,
        "X_test":            X_test,
        "y_test_actual":     y_test_actual,
        "y_test_predicted":  y_test_predicted,
    }


def print_feature_importance(rf: RandomForestRegressor, feature_names, top_n: int = TOP_N_FEATURES):
    """Print and plot the top N features by mean decrease in impurity (MDI)."""
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n)

    print(f"\nTop {top_n} most important features (mean decrease in impurity):")
    print("─" * 52)
    for rank, (feat, imp) in enumerate(top.items(), start=1):
        print(f"  {rank:2d}.  {feat:<40s}  {imp:.5f}")

    _, ax = plt.subplots(figsize=(10, 6))
    top.iloc[::-1].plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances — RF Aggregation GDP Nowcast")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out_path = THIS_DIR / "feature_importances_aggre.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved to {out_path}")


# =============================================================================
# FULL PIPELINE
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("Step 1: Load filled data from Supabase")
    print("=" * 55)
    df_md_filled, df_qd_filled = load_filled_data()

    print("=" * 55)
    print("Step 2: Simple aggregation feature engineering")
    print("=" * 55)
    X, y = build_feature_matrix(df_md_filled, df_qd_filled)

    print("X — first 5 rows (first 3 cols):")
    print(X.iloc[:5, :3].to_string())
    print("\ny — first 5 rows:")
    print(y.head().to_string())
    print()

    print("=" * 55)
    print("Step 3: Train/Val/Test split + CV + RF fitting")
    print("=" * 55)
    X_trainval, X_test, y_trainval, y_test = split_data(X, y)
    best_mf  = tune_max_features(X_trainval, y_trainval)
    rf_final = fit_final_model(X_trainval, y_trainval, best_mf)

    print("=" * 55)
    print("Step 4: Feature importance (evaluation via evaluation.py)")
    print("=" * 55)
    print_feature_importance(rf_final, X.columns)
