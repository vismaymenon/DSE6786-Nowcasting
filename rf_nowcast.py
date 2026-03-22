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
"""

import warnings
warnings.filterwarnings("ignore")

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Config ─────────────────────────────────────────────────────────────────────
FRED_API_KEY   = "4c68fd37456d1706c93321651fd0efa5"
SERIES_ID      = "A191RL1Q225SBEA"   # Real GDP growth, quarterly, SAAR
N_LAGS         = 4                   # AR features: lag_1 … lag_4
TEST_FRAC      = 0.10                # last 10% of observations → test set
N_SPLITS_CV    = 5                   # time-series CV folds for max_features search
N_TREES_SEARCH = 200                 # trees during CV search (speed)
N_TREES_FINAL  = 1000                # trees for final published model
MAX_SAMPLES    = 0.8                 # bootstrap sample rate (reduces tree correlation)
RANDOM_STATE   = 42

FEATURE_COLS = [f"lag_{i}" for i in range(1, N_LAGS + 1)]
CACHE_FILE   = f"data_{SERIES_ID}.csv"   # local cache for GDP growth series

# ── 1. Load GDP growth (cache → FRED) ─────────────────────────────────────────
import os

if os.path.exists(CACHE_FILE):
    print(f"Loading GDP growth data from local cache ({CACHE_FILE}) …")
    gdp = pd.read_csv(CACHE_FILE, index_col="date", parse_dates=True)
else:
    print("Cache not found — fetching GDP growth data from FRED …")
    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": SERIES_ID, "api_key": FRED_API_KEY, "file_type": "json"},
    )
    resp.raise_for_status()

    raw = pd.DataFrame(resp.json()["observations"])
    raw["date"]  = pd.to_datetime(raw["date"])
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw.set_index("date", inplace=True)

    gdp = raw[["value"]].rename(columns={"value": "gdp_growth"}).dropna().sort_index()
    gdp.to_csv(CACHE_FILE)
    print(f"  Saved to {CACHE_FILE}")

print(f"  Raw series : {len(gdp)} quarters  "
      f"({gdp.index[0].date()} → {gdp.index[-1].date()})")

# ── 2. Build autoregressive features ──────────────────────────────────────────
for lag in range(1, N_LAGS + 1):
    gdp[f"lag_{lag}"] = gdp["gdp_growth"].shift(lag)

gdp.dropna(inplace=True)
print(f"  After lags : {len(gdp)} quarters  "
      f"({gdp.index[0].date()} → {gdp.index[-1].date()})")

# ── 3. Temporal train / test split (90 / 10) ──────────────────────────────────
n_total    = len(gdp)
n_test     = max(1, int(np.round(n_total * TEST_FRAC)))
n_train    = n_total - n_test
split_idx  = n_train

X = gdp[FEATURE_COLS].values
y = gdp["gdp_growth"].values
dates = gdp.index

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_train = dates[:split_idx]
dates_test  = dates[split_idx:]

print(f"\nTrain : {n_train} obs  ({dates_train[0].date()} → {dates_train[-1].date()})")
print(f"Test  : {n_test}  obs  ({dates_test[0].date()}  → {dates_test[-1].date()})")
print(f"Test share : {n_test / n_total * 100:.1f}%")

# ── 4. Tune max_features via 5-fold TimeSeriesSplit Expanidng Window CV on training set ─────────
print(f"\nTuning max_features ∈ {{1, 2, 3, 4}} via {N_SPLITS_CV}-fold time-series CV\n")

tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
mtry_candidates = list(range(1, N_LAGS + 1))
mtry_cv_rmse    = {}

for mtry in mtry_candidates:
    fold_rmses = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        rf_tmp = RandomForestRegressor(
            n_estimators=N_TREES_SEARCH,
            max_features=mtry,
            max_samples=MAX_SAMPLES,
            random_state=RANDOM_STATE,
        )
        rf_tmp.fit(X_train[tr_idx], y_train[tr_idx])
        pred = rf_tmp.predict(X_train[val_idx])
        fold_rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], pred)))

    mtry_cv_rmse[mtry] = np.mean(fold_rmses)
    print(f"  max_features={mtry}  |  CV RMSE={mtry_cv_rmse[mtry]:.4f}")

best_mtry = min(mtry_cv_rmse, key=mtry_cv_rmse.get)
print(f"\n>>> Best max_features = {best_mtry}  (CV RMSE = {mtry_cv_rmse[best_mtry]:.4f})")

# ── 5. Final CV with best max_features (report per-fold metrics) ───────────────
print(f"\nFinal CV with max_features={best_mtry}, n_estimators={N_TREES_SEARCH} …\n")
cv_rmse, cv_mae = [], []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
    rf_cv = RandomForestRegressor(
        n_estimators=N_TREES_SEARCH,
        max_features=best_mtry,
        max_samples=MAX_SAMPLES,
        random_state=RANDOM_STATE,
    )
    rf_cv.fit(X_train[tr_idx], y_train[tr_idx])
    pred_val = rf_cv.predict(X_train[val_idx])

    rmse = np.sqrt(mean_squared_error(y_train[val_idx], pred_val))
    mae  = mean_absolute_error(y_train[val_idx], pred_val)
    cv_rmse.append(rmse)
    cv_mae.append(mae)
    print(f"  Fold {fold}  |  train={len(tr_idx):3d}  val={len(val_idx):3d}  "
          f"|  RMSE={rmse:.4f}  MAE={mae:.4f}")

print(f"\nCV mean RMSE : {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")
print(f"CV mean MAE  : {np.mean(cv_mae):.4f} ± {np.std(cv_mae):.4f}")

# ── 6. Fit final model on full training set ────────────────────────────────────
print(f"\nFitting final model  (n_estimators={N_TREES_FINAL}, "
      f"max_features={best_mtry}, max_samples={MAX_SAMPLES}) …")

rf_final = RandomForestRegressor(
    n_estimators=N_TREES_FINAL,
    max_features=best_mtry,
    max_samples=MAX_SAMPLES,
    oob_score=True,
    random_state=RANDOM_STATE,
)
rf_final.fit(X_train, y_train)

oob_rmse     = np.sqrt(mean_squared_error(y_train, rf_final.oob_prediction_))
y_pred_test  = rf_final.predict(X_test)
test_rmse    = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae     = mean_absolute_error(y_test, y_pred_test)

print("\n" + "─" * 45)
print(f"{'OOB  RMSE (train sanity check)':35s}: {oob_rmse:.4f}")
print(f"{'Out-of-sample RMSE':35s}: {test_rmse:.4f}")
print(f"{'Out-of-sample MAE':35s}: {test_mae:.4f}")
print("─" * 45)

# ── 7. Feature importances ─────────────────────────────────────────────────────
importances = pd.Series(rf_final.feature_importances_, index=FEATURE_COLS)
print("\nFeature importances:")
for feat, imp in importances.sort_values(ascending=False).items():
    print(f"  {feat}: {imp:.4f}")

# ── 8. Plots ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 11))

# (a) Full series — train + test predictions
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
ax3.bar(
    [str(m) for m in mtry_candidates],
    [mtry_cv_rmse[m] for m in mtry_candidates],
    color=["firebrick" if m == best_mtry else "steelblue" for m in mtry_candidates],
    edgecolor="white",
)
ax3.set_title(f"max_features Tuning — CV RMSE  (best={best_mtry})")
ax3.set_xlabel("max_features (m_try)")
ax3.set_ylabel("Mean CV RMSE")
ax3.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("rf_nowcast_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to rf_nowcast_results.png")