import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pipeline.load_data as load_data
import pipeline.poos as poos
import pipeline.models.autoregressive as autoregressive
import pipeline.models.rf_benchmark as rf_benchmark

AR_LAGS    = 2
RF_LAGS    = 4
PROP_TRAIN = 0.9

def build_features(gdp, n_lags):
    df = gdp.rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)
    df.dropna(inplace=True)
    X = df[[f"lag_{i}" for i in range(1, n_lags + 1)]]
    y = df["gdp_growth"]
    return X, y

# ── Load data ─────────────────────────────────────────────────────────────────
gdp = load_data.load_gdp()

# ── AR(2) POOS ────────────────────────────────────────────────────────────────
print("\n=== Autoregressive Model AR(2) ===")
X_ar, y_ar = build_features(gdp, AR_LAGS)
_, ar_out, ar_rmse, ar_mae = poos.poos_validation(
    method=autoregressive.ar_model_nowcast,
    X=X_ar,
    y=y_ar,
    prop_train=PROP_TRAIN,
)

# ── RF POOS ────────────────────────────────────────────────────────────────
print("\n=== Random Forest Model RF ===")
X_rf, y_rf = build_features(gdp, RF_LAGS)
_, rf_out, rf_rmse, rf_mae = poos.poos_validation(
    method=rf_benchmark.rf_model_nowcast,
    X=X_rf,
    y=y_rf,
    prop_train=PROP_TRAIN,
)

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 45)
print(f"{'Model':<20} {'RMSE':>10} {'MAE':>10}")
print("─" * 45)
print(f"{'Autoregressive':<20} {ar_rmse:>10.4f} {ar_mae:>10.4f}")
print(f"{'Random Forest':<20} {rf_rmse:>10.4f} {rf_mae:>10.4f}")
print("=" * 45)
print(f"\nOOS observations: {len(ar_out)}")

# ── Plots ─────────────────────────────────────────────────────────────────────
poos.plot_poos_results(y_ar, ar_out, title="Autoregressive Model AR(2) POOS")
poos.plot_poos_results(y_rf, rf_out, title="Random Forest Model RF POOS")
