"""
rf_AR.py
========
Final Random Forest nowcasting model for GDP growth.

Pipeline
--------
  Step 1 — Lag selection
            Execute ragged_edge_working.ipynb via nbconvert.
            The notebook selects the optimal AR(p) per variable (BIC criterion)
            for fred_md and fred_qd_X, then writes the results to bic_lags.csv
            in this directory (pipeline/models/).

  Step 2 — Ragged-edge imputation
            Call fill_ragged_edge() from pipeline/ragged_edge.py.
            For each variable that has trailing NaNs (unreleased recent
            observations), it fits an AR(p) model on the observed history
            and forecasts those missing values one step ahead.

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
import subprocess
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

NOTEBOOK_PATH = THIS_DIR / "ragged_edge_working.ipynb"
BIC_LAGS_PATH = THIS_DIR / "bic_lags.csv"

# Allow importing from pipeline/
sys.path.insert(0, str(PIPELINE_DIR))
from ragged_edge import fill_ragged_edge


# =============================================================================
# STEP 1 — Execute notebook to get optimal AR lags → bic_lags.csv
# =============================================================================

def run_lag_selection_notebook():
    """
    Run ragged_edge_working.ipynb in-place using nbconvert.

    The notebook:
      1. Reads fred_md.csv and fred_qd_X.csv (filtered to 1959–2019 for stability)
      2. Selects the optimal AR lag p for each variable by minimising BIC
      3. Saves the results as bic_lags.csv in the same directory (pipeline/models/)

    We run it with cwd=THIS_DIR so that relative paths inside the notebook
    (e.g. pd.read_csv("../../data/fred_md.csv")) resolve correctly.
    """
    print("Step 1 — Running ragged_edge_working.ipynb to select optimal AR lags …")

    result = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=600",   # 10-min timeout (lag selection is slow)
            "--output", str(NOTEBOOK_PATH),         # overwrite the notebook with output
            str(NOTEBOOK_PATH),
        ],
        capture_output=True,
        text=True,
        cwd=str(THIS_DIR),   # run from notebook's directory so its relative paths work
    )

    if result.returncode != 0:
        print("nbconvert stderr:\n", result.stderr)
        raise RuntimeError(
            "Notebook execution failed. Make sure 'jupyter' is installed and "
            "the notebook kernel is available."
        )

    if not BIC_LAGS_PATH.exists():
        raise FileNotFoundError(
            f"{BIC_LAGS_PATH} was not found after notebook execution. "
            "Check that the notebook's last cell writes bic_lags.csv."
        )

    lag_df = pd.read_csv(BIC_LAGS_PATH)
    print(f"  Optimal lags selected for {len(lag_df)} variables.")
    print(f"  Saved to {BIC_LAGS_PATH}\n")


# =============================================================================
# STEP 2 — Ragged-edge imputation using pipeline/ragged_edge.py
# =============================================================================

def run_imputation():
    """
    Call fill_ragged_edge() for fred_md and fred_qd_X.

    fill_ragged_edge() (defined in pipeline/ragged_edge.py):
      - Reads a data CSV and a lag CSV (bic_lags.csv)
      - For each variable that has trailing NaNs, fits an AR(p) model on its
        observed history and fills the missing values by one-step-ahead forecasting
      - Returns the filled DataFrame

    Quarterly data pre-processing note:
        fred_qd_X.csv has 2 rows with NaN dates — these are empty placeholder
        rows for future quarters that haven't started yet.  We drop them before
        imputation so fill_ragged_edge() doesn't treat them as real observations
        and forecast 2 phantom extra quarters.

    Returns
    -------
    df_md_filled  : pd.DataFrame — monthly FRED-MD with ragged edge filled
    df_qd_filled  : pd.DataFrame — quarterly FRED-QD-X with ragged edge filled
    """
    print("Step 2 — Ragged-edge imputation …\n")

    # ── Monthly (fred_md) ─────────────────────────────────────────────────────
    # fred_md has no NaN-date rows, so we pass it directly.
    print("  Imputing fred_md …")
    df_md_filled = fill_ragged_edge(
        data_csv=str(DATA_DIR / "fred_md.csv"),
        lag_csv=str(BIC_LAGS_PATH),
    )
    md_nans_remaining = df_md_filled.drop(columns=["sasdate"]).isna().sum().sum()
    print(f"  Done. {md_nans_remaining} NaNs remaining in fred_md.\n")

    # ── Quarterly (fred_qd_X) ─────────────────────────────────────────────────
    # Drop the 2 NaN-date placeholder rows before imputing.
    print("  Imputing fred_qd_X …")
    df_qd_raw = pd.read_csv(DATA_DIR / "fred_qd_X.csv", parse_dates=["sasdate"])
    df_qd_raw = df_qd_raw.dropna(subset=["sasdate"]).reset_index(drop=True)

    # Write to a temporary CSV so fill_ragged_edge() can read it by path
    _tmp_qd = THIS_DIR / "_tmp_fred_qd.csv"
    df_qd_raw.to_csv(_tmp_qd, index=False)

    df_qd_filled = fill_ragged_edge(
        data_csv=str(_tmp_qd),
        lag_csv=str(BIC_LAGS_PATH),
    )
    _tmp_qd.unlink()   # clean up temp file

    qd_nans_remaining = df_qd_filled.drop(columns=["sasdate"]).isna().sum().sum()
    print(f"  Done. {qd_nans_remaining} NaNs remaining in fred_qd_X.\n")

    return df_md_filled, df_qd_filled


# =============================================================================
# MAIN ENTRY POINT FOR STEPS 1–2
# =============================================================================

def load_and_impute():
    """
    Run Steps 1–2 end-to-end.

    Returns
    -------
    df_md_filled : pd.DataFrame — monthly FRED-MD, ragged edge imputed
    df_qd_filled : pd.DataFrame — quarterly FRED-QD-X, ragged edge imputed
    """
    run_lag_selection_notebook()
    df_md_filled, df_qd_filled = run_imputation()

    # ── Quick sanity check: show the last 3 rows of each dataset ──────────────
    print("fred_md tail (first 3 feature columns):")
    print(df_md_filled.tail(3)[["sasdate"] + list(df_md_filled.columns[1:4])].to_string())
    print()
    print("fred_qd_X tail (first 3 feature columns):")
    print(df_qd_filled.tail(3)[["sasdate"] + list(df_qd_filled.columns[1:4])].to_string())

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
# STEP 5 — EVALUATION + FEATURE IMPORTANCE
# =============================================================================

def evaluate(rf: RandomForestRegressor, X_test, y_test, y_trainval) -> dict:
    """Report OOB RMSE (train+val sanity check) and held-out test RMSE / MAE."""
    oob_rmse  = np.sqrt(mean_squared_error(y_trainval.values, rf.oob_prediction_))
    y_pred    = rf.predict(X_test.values)
    test_rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
    test_mae  = mean_absolute_error(y_test.values, y_pred)

    print("\n" + "─" * 52)
    print(f"{'OOB RMSE  (train+val sanity check)':42s}: {oob_rmse:.4f}")
    print(f"{'Test RMSE (held-out 100 quarters)':42s}: {test_rmse:.4f}")
    print(f"{'Test MAE  (held-out 100 quarters)':42s}: {test_mae:.4f}")
    print("─" * 52)

    return {"y_pred": y_pred, "test_rmse": test_rmse, "test_mae": test_mae, "oob_rmse": oob_rmse}


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
    print("Steps 1–2: Lag selection + Ragged-edge imputation")
    print("=" * 55)
    df_md_filled, df_qd_filled = load_and_impute()

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

    # ── Step 5: Evaluate + feature importance ─────────────────────────────────
    print("=" * 55)
    print("Step 5: Evaluation + Feature importance")
    print("=" * 55)
    results = evaluate(rf_final, X_test, y_test, y_trainval)
    print_feature_importance(rf_final, X.columns)
