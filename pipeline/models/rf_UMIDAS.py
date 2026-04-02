"""
rf_AR.py
========
Final Random Forest nowcasting model for GDP growth.

Pipeline
--------
  Step 1 — Load filled data
            Fetch ragged-edge-imputed data from Supabase (filled_md, filled_qd).
            Run test.py beforehand to populate these tables from the latest
            FRED release.

  Step 3 — U-MIDAS feature engineering
            Each monthly variable becomes 3 quarterly features (_m1, _m2, _m3)
            rather than being averaged, preserving within-quarter dynamics.
            115 monthly × 3 = 345  +  206 quarterly  =  551 features total.

  Step 4 — Random Forest with expanding-window CV
            TimeSeriesSplit (6 folds) tunes max_features only.
            max_depth=None and min_samples_leaf=1 (sklearn defaults).

  Step 5 — Evaluation and feature importance

Data split (266 quarterly GDP observations):
  Train     : obs   0–99   (100 quarters)  ← initial CV window
  Validation: obs 100–165  ( 66 quarters)  ← CV expands through here
  Test      : obs 166–265  (100 quarters)  ← strictly held out
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
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── RF config ─────────────────────────────────────────────────────────────────
N_TRAIN           = 100
N_TRAINVAL        = 166     # 100 train + 66 val
N_SPLITS_CV       = 6       # expanding-window CV folds
VAL_FOLD_SIZE     = 11      # quarters per fold  (66 / 6 = 11)
N_TREES_CV        = 300     # trees during CV (speed)
N_TREES_FINAL     = 1000    # trees for the final model
MAX_FEATURES_GRID = ["sqrt", "log2", 0.1, 0.3]
TOP_N_FEATURES    = 20
RANDOM_STATE      = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent   # pipeline/models/
PIPELINE_DIR = THIS_DIR.parent                   # pipeline/
PROJECT_DIR  = PIPELINE_DIR.parent               # project root
DATA_DIR     = PROJECT_DIR / "data"

# Allow importing from pipeline/ and project root
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

    These tables are populated by running test.py, which calls fill_ragged_edge()
    on the latest FRED release and upserts the results.

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
# STEP 3 — U-MIDAS FEATURE ENGINEERING
# =============================================================================

def build_umidas_features(df_md_filled: pd.DataFrame) -> pd.DataFrame:
    """
    Convert filled monthly data into U-MIDAS quarterly features.

    For each quarter Q, the 3 monthly observations inside Q are kept as
    separate columns (_m1 oldest → _m3 most recent) instead of being averaged.

    Quarter label convention (matches gdp.csv): first day of the last month in
    the quarter — e.g. Q1 1959 → 1959-03-01, Q2 1959 → 1959-06-01.
    Computed as: period_start + 2 months
        Jan 1959 → Q1 start 1959-01-01 + 2 months = 1959-03-01 ✓

    Returns DataFrame indexed by quarter label, shape (n_quarters, n_vars × 3).
    """
    df = df_md_filled.copy().sort_values("sasdate").reset_index(drop=True)

    # Map every monthly row to its quarter label
    df["qtr_label"] = (
        df["sasdate"].dt.to_period("Q").dt.start_time
        + pd.DateOffset(months=2)
    )

    # Position within the quarter: 1 = oldest month, 3 = most recent
    df["month_pos"] = df.groupby("qtr_label").cumcount() + 1

    feature_cols = [c for c in df.columns if c not in ("sasdate", "qtr_label", "month_pos")]
    
    # Drop COVID flag columns before U-MIDAS (they're kept as quarterly features)
    feature_cols = [c for c in df.columns 
                if c not in ("sasdate", "qtr_label", "month_pos",
                             "covid_crash", "covid_recover")]

    # Pivot to wide format: rows = quarters, columns = (variable, month_pos)
    df_pivot = df.pivot(index="qtr_label", columns="month_pos", values=feature_cols)

    # Flatten MultiIndex: ('RPI_t', 1) → 'RPI_t_m1'
    df_pivot.columns = [f"{col}_m{pos}" for col, pos in df_pivot.columns]
    df_pivot.index.name = "sasdate"

    return df_pivot


def build_feature_matrix(df_md_filled: pd.DataFrame, df_qd_filled: pd.DataFrame) -> tuple:
    """
    Assemble the full quarterly feature matrix X and GDP target y.

    Joins U-MIDAS monthly features (345 cols) and quarterly FRED-QD-X features
    (206 cols), aligned to the 266 GDP quarter dates.

    Returns
    -------
    X : pd.DataFrame  shape (266, 551)
    y : pd.Series     shape (266,)
    """
    gdp = pd.read_csv(DATA_DIR / "gdp.csv", parse_dates=["sasdate"])
    gdp = gdp.set_index("sasdate").sort_index()

    df_umidas = build_umidas_features(df_md_filled)

    df_qd = df_qd_filled.copy()
    df_qd["sasdate"] = pd.to_datetime(df_qd["sasdate"])
    df_qd = df_qd.set_index("sasdate").sort_index()

    # Inner join on quarter dates, then trim to GDP index (drops Q3 2025 — no GDP yet)
    X = df_umidas.join(df_qd, how="inner").reindex(gdp.index)
    y = gdp["GDPC1_t"]

    # Drop rows where any NaN remains (safety check after imputation)
    valid = X.notna().all(axis=1) & y.notna()
    if (~valid).sum() > 0:
        print(f"  Warning: dropping {(~valid).sum()} rows with remaining NaNs.")
        X, y = X[valid], y[valid]

    print(f"Feature matrix: {X.shape[0]} quarters × {X.shape[1]} features")
    print(f"  U-MIDAS monthly  : {df_umidas.shape[1]} features")
    print(f"  Quarterly (QD-X) : {df_qd.shape[1]} features\n")

    return X, y


# =============================================================================
# STEP 4 — SPLIT + CROSS-VALIDATION + FITTING
# =============================================================================

def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Strict chronological split: train (100) / validation (66) / test (100).
    X_trainval (166 obs) is used for CV. X_test is never touched until final eval.
    """
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
    """
    Select best max_features using 6-fold expanding-window TimeSeriesSplit CV.

    Fold structure over the 166 train+val observations:
    ──────────────────────────────────────────────────────────────────
    Fold 1: train [  0:100]  →  val [100:111]   ← first fold starts at N_TRAIN
    Fold 2: train [  0:111]  →  val [111:122]
    Fold 3: train [  0:122]  →  val [122:133]
    Fold 4: train [  0:133]  →  val [133:144]
    Fold 5: train [  0:144]  →  val [144:155]
    Fold 6: train [  0:155]  →  val [155:166]
    ──────────────────────────────────────────────────────────────────
    test_size=11 forces each fold's val window to 11 quarters,
    which makes the first fold's train window exactly 166 − 6×11 = 100 obs.

    Candidates and what they mean with 551 features:
      "sqrt" → √551 ≈ 23 features per split
      "log2" → log₂(551) ≈ 9 features per split
      0.1    → 55 features per split
      0.3    → 165 features per split
    """
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
        return float(best_key)       # e.g. "0.1" → 0.1
    except ValueError:
        return best_key              # "sqrt" or "log2" — keep as string


def fit_final_model(X_trainval: pd.DataFrame, y_trainval: pd.Series, best_mf) -> RandomForestRegressor:
    """
    Refit on the full train+val set (166 obs) with the CV-selected max_features.

    Hyperparameters:
      max_features     = best_mf from CV
      max_depth        = None  (fully grown trees; sklearn default)
      min_samples_leaf = 1     (sklearn default)
      n_estimators     = 1000  (more trees → lower variance than CV's 300)
      max_samples      = 0.8   (bootstrap uses 80% of training rows per tree)
      oob_score        = True  (free out-of-bag error estimate on train+val)
    """
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
# STEP 5 — POOS-COMPATIBLE MODEL WRAPPER
# =============================================================================

def rf_umidas_nowcast(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    POOS-compatible wrapper for the RF U-MIDAS model.

    Follows the contract expected by poos.poos_validation():
      - Last row of X / last element of y is the test observation.
      - All preceding rows are used for training.
      - Returns a dict with keys: X_train, y_train, y_train_predicted,
        X_test, y_test_actual, y_test_predicted.

    Uses fixed max_features="sqrt" (best CV choice) and N_TREES_CV trees
    to keep POOS runtime practical — CV tuning is not repeated per window.
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
    """
    Print and plot the top N features by mean decrease in impurity (MDI).
    MDI measures how much each feature reduces variance across all trees —
    higher = used more frequently at high-impact splits.
    """
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n)

    print(f"\nTop {top_n} most important features (mean decrease in impurity):")
    print("─" * 52)
    for rank, (feat, imp) in enumerate(top.items(), start=1):
        print(f"  {rank:2d}.  {feat:<40s}  {imp:.5f}")

    _, ax = plt.subplots(figsize=(10, 6))
    top.iloc[::-1].plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest GDP Nowcast")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out_path = THIS_DIR / "feature_importances.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved to {out_path}")


# =============================================================================
# FULL PIPELINE
# =============================================================================

if __name__ == "__main__":
    # ── Steps 1–2: lag selection + ragged-edge imputation ────────────────────
    print("=" * 55)
    print("Step 1: Load filled data from Supabase")
    print("=" * 55)
    df_md_filled, df_qd_filled = load_filled_data()

    # ── Step 3: U-MIDAS feature engineering ──────────────────────────────────
    print("=" * 55)
    print("Step 3: U-MIDAS feature engineering")
    print("=" * 55)
    X, y = build_feature_matrix(df_md_filled, df_qd_filled)

    # ── Step 4: Split + CV + fit ──────────────────────────────────────────────
    print("=" * 55)
    print("Step 4: Train/Val/Test split + CV + RF fitting")
    print("=" * 55)
    X_trainval, X_test, y_trainval, y_test = split_data(X, y)
    best_mf  = tune_max_features(X_trainval, y_trainval)
    rf_final = fit_final_model(X_trainval, y_trainval, best_mf)

    # ── Step 5: Feature importance (evaluation via evaluation.py) ────────────
    print("=" * 55)
    print("Step 5: Feature importance")
    print("=" * 55)
    print_feature_importance(rf_final, X.columns)
