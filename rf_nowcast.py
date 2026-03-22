"""
rf_nowcast.py
=============
Random Forest benchmark for GDP growth nowcasting.

Design choices (per Breiman 2001, Probst et al. 2019, Goulet Coulombe 2020):
  - Features   : 4 autoregressive lags of GDP growth (lag_1 … lag_4)
  - Train/test : 90 / 10 % temporal split (no shuffling)
  - CV         : 5-fold TimeSeriesSplit (expanding window) on training set (not KFold to prevent data leakage)
  - max_features: tuned over {1, 2, 3, 4} via CV (n_estimators=200 during search)
    - mtry: number of features to consider when looking for the best split at each node of a decision tree
  - Final model: n_estimators=1000, max_samples=0.8, oob_score=True, max_depth=None (fully grown tree), min_samples_leaf=1, random_state=42

Usage (import):
    from rf_nowcast import fit_rf_nowcast
    results = fit_rf_nowcast(gdp_series)

Usage (standalone):
    python rf_nowcast.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Default config ──────────────────────────────────────────────────────────────
FRED_API_KEY   = "4c68fd37456d1706c93321651fd0efa5"
SERIES_ID      = "A191RL1Q225SBEA"
N_LAGS         = 4
TEST_FRAC      = 0.10
N_SPLITS_CV    = 5
N_TREES_SEARCH = 200
N_TREES_FINAL  = 1000
MAX_SAMPLES    = 0.8
RANDOM_STATE   = 42


def load_gdp(api_key=FRED_API_KEY, series_id=SERIES_ID, cache_file=None):
    """
    Load GDP growth series from local cache or FRED API.

    Returns
    -------
    pd.Series  — quarterly GDP growth with DatetimeIndex
    """
    if cache_file is None:
        cache_file = f"data_{series_id}.csv"

    if os.path.exists(cache_file):
        print(f"Loading GDP growth from cache ({cache_file}) …")
        gdp = pd.read_csv(cache_file, index_col="date", parse_dates=True).squeeze()
    else:
        print("Cache not found — fetching from FRED …")
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
        )
        resp.raise_for_status()
        raw = pd.DataFrame(resp.json()["observations"])
        raw["date"]  = pd.to_datetime(raw["date"])
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw.set_index("date", inplace=True)
        gdp = raw["value"].rename("gdp_growth").dropna().sort_index()
        gdp.to_csv(cache_file, index_label="date")
        print(f"  Saved to {cache_file}")

    print(f"  {len(gdp)} quarters  ({gdp.index[0].date()} → {gdp.index[-1].date()})")
    return gdp


def fit_rf_nowcast(
    gdp_series,
    n_lags         = N_LAGS,
    test_frac      = TEST_FRAC,
    n_splits_cv    = N_SPLITS_CV,
    n_trees_search = N_TREES_SEARCH,
    n_trees_final  = N_TREES_FINAL,
    max_samples    = MAX_SAMPLES,
    random_state   = RANDOM_STATE,
    verbose        = True,
):
    """
    Fit a Random Forest nowcasting model on a GDP growth series.

    Parameters
    ----------
    gdp_series     : pd.Series  — quarterly GDP growth with DatetimeIndex
    n_lags         : int        — number of AR lags to use as features (default 4)
    test_frac      : float      — fraction of data held out as test set (default 0.10)
    n_splits_cv    : int        — number of TimeSeriesSplit folds for max_features tuning
    n_trees_search : int        — trees used during CV search (speed)
    n_trees_final  : int        — trees for the final fitted model
    max_samples    : float      — bootstrap sample rate per tree (0 < max_samples ≤ 1)
    random_state   : int        — random seed
    verbose        : bool       — print progress

    Returns
    -------
    dict with keys:
        model             : fitted RandomForestRegressor
        best_mtry         : best max_features found by CV
        feature_cols      : list of feature column names
        X_train, y_train  : training arrays
        X_test,  y_test   : test arrays
        dates_train       : DatetimeIndex for training set
        dates_test        : DatetimeIndex for test set
        y_pred_test       : predicted GDP growth on test set
        oob_rmse          : OOB RMSE on training set (sanity check)
        test_rmse         : out-of-sample RMSE
        test_mae          : out-of-sample MAE
        mtry_cv_rmse      : dict {max_features: mean CV RMSE} from tuning
        feature_importances: pd.Series of impurity-based feature importances
    """
    # ── 1. Build AR features ───────────────────────────────────────────────────
    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    df = gdp_series.rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)
    df.dropna(inplace=True)

    # ── 2. Temporal train / test split ─────────────────────────────────────────
    n_total   = len(df)
    n_test    = max(1, int(np.round(n_total * test_frac)))
    n_train   = n_total - n_test
    split_idx = n_train

    X = df[feature_cols].values
    y = df["gdp_growth"].values
    dates = df.index

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train = dates[:split_idx]
    dates_test  = dates[split_idx:]

    if verbose:
        print(f"\nTrain : {n_train} obs  ({dates_train[0].date()} → {dates_train[-1].date()})")
        print(f"Test  : {n_test}  obs  ({dates_test[0].date()}  → {dates_test[-1].date()})")

    # ── 3. Tune max_features via TimeSeriesSplit CV ────────────────────────────
    if verbose:
        print(f"\nTuning max_features ∈ {{1 … {n_lags}}} via {n_splits_cv}-fold CV …\n")

    tscv = TimeSeriesSplit(n_splits=n_splits_cv)
    mtry_cv_rmse = {}

    for mtry in range(1, n_lags + 1):
        fold_rmses = []
        for tr_idx, val_idx in tscv.split(X_train):
            rf_tmp = RandomForestRegressor(
                n_estimators=n_trees_search,
                max_features=mtry,
                max_samples=max_samples,
                random_state=random_state,
            )
            rf_tmp.fit(X_train[tr_idx], y_train[tr_idx])
            pred = rf_tmp.predict(X_train[val_idx])
            fold_rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], pred)))
        mtry_cv_rmse[mtry] = np.mean(fold_rmses)
        if verbose:
            print(f"  max_features={mtry}  |  CV RMSE={mtry_cv_rmse[mtry]:.4f}")

    best_mtry = min(mtry_cv_rmse, key=mtry_cv_rmse.get)
    if verbose:
        print(f"\n>>> Best max_features = {best_mtry}  (CV RMSE = {mtry_cv_rmse[best_mtry]:.4f})")

    # ── 4. Fit final model ─────────────────────────────────────────────────────
    if verbose:
        print(f"\nFitting final model (n_estimators={n_trees_final}, "
              f"max_features={best_mtry}, max_samples={max_samples}) …")

    rf_final = RandomForestRegressor(
        n_estimators=n_trees_final,
        max_features=best_mtry,
        max_samples=max_samples,
        oob_score=True,
        random_state=random_state,
    )
    rf_final.fit(X_train, y_train)

    oob_rmse    = np.sqrt(mean_squared_error(y_train, rf_final.oob_prediction_))
    y_pred_test = rf_final.predict(X_test)
    test_rmse   = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae    = mean_absolute_error(y_test, y_pred_test)

    if verbose:
        print("\n" + "─" * 45)
        print(f"{'OOB  RMSE (train sanity check)':35s}: {oob_rmse:.4f}")
        print(f"{'Out-of-sample RMSE':35s}: {test_rmse:.4f}")
        print(f"{'Out-of-sample MAE':35s}: {test_mae:.4f}")
        print("─" * 45)

    feature_importances = pd.Series(rf_final.feature_importances_, index=feature_cols)

    return {
        "model":               rf_final,
        "best_mtry":           best_mtry,
        "feature_cols":        feature_cols,
        "X_train":             X_train,
        "y_train":             y_train,
        "X_test":              X_test,
        "y_test":              y_test,
        "dates_train":         dates_train,
        "dates_test":          dates_test,
        "y_pred_test":         y_pred_test,
        "oob_rmse":            oob_rmse,
        "test_rmse":           test_rmse,
        "test_mae":            test_mae,
        "mtry_cv_rmse":        mtry_cv_rmse,
        "feature_importances": feature_importances,
    }


def plot_rf_results(results, save_path="rf_nowcast_results.png"):
    """
    Plot predictions and max_features tuning from fit_rf_nowcast() results dict.
    """
    dates_train   = results["dates_train"]
    dates_test    = results["dates_test"]
    y_train       = results["y_train"]
    y_test        = results["y_test"]
    y_pred_test   = results["y_pred_test"]
    mtry_cv_rmse  = results["mtry_cv_rmse"]
    best_mtry     = results["best_mtry"]
    test_rmse     = results["test_rmse"]
    test_mae      = results["test_mae"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    # (a) Full series
    ax = axes[0]
    ax.plot(dates_train, y_train, label="Train (actual)", color="steelblue", linewidth=1.2)
    ax.plot(dates_test, y_test, label="Test (actual)", color="darkorange", linewidth=1.5)
    ax.plot(dates_test, y_pred_test, label="Test (predicted)", color="firebrick",
            linestyle="--", marker="o", markersize=4, linewidth=1.2)
    ax.axvline(dates_test[0], color="grey", linestyle=":", linewidth=1.5, label="Train / Test split")
    ax.set_title("Random Forest — GDP Growth Nowcast (AR-4)")
    ax.set_ylabel("GDP Growth (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Test set zoom
    ax2 = axes[1]
    ax2.plot(dates_test, y_test, label="Actual", color="darkorange", linewidth=1.5, marker="o", markersize=4)
    ax2.plot(dates_test, y_pred_test, label="Predicted", color="firebrick",
             linestyle="--", marker="s", markersize=4, linewidth=1.2)
    ax2.fill_between(dates_test, y_test, y_pred_test, alpha=0.15, color="grey")
    ax2.set_title(f"Test Set — Actual vs Predicted  (RMSE={test_rmse:.3f}, MAE={test_mae:.3f})")
    ax2.set_ylabel("GDP Growth (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # (c) max_features CV RMSE
    ax3 = axes[2]
    mtry_keys = list(mtry_cv_rmse.keys())
    ax3.bar(
        [str(m) for m in mtry_keys],
        [mtry_cv_rmse[m] for m in mtry_keys],
        color=["firebrick" if m == best_mtry else "steelblue" for m in mtry_keys],
        edgecolor="white",
    )
    ax3.set_title(f"max_features Tuning — CV RMSE  (best={best_mtry})")
    ax3.set_xlabel("max_features (m_try)")
    ax3.set_ylabel("Mean CV RMSE")
    ax3.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to {save_path}")


# ── Standalone entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    gdp     = load_gdp()
    results = fit_rf_nowcast(gdp)
    plot_rf_results(results)
