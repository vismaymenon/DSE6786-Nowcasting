import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pipeline.load_data as load_data
import pipeline.poos as poos
import pipeline.models.autoregressive as autoregressive
import pipeline.models.rf_benchmark as rf_benchmark

N_LAGS     = 4
PROP_TRAIN = 0.9

# ── Load data ─────────────────────────────────────────────────────────────────
gdp = load_data.load_gdp()

# ── Build AR lag features ─────────────────────────────────────────────────────
df = gdp.rename("gdp_growth").to_frame()
for lag in range(1, N_LAGS + 1):
    df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)
df.dropna(inplace=True)

X = df[[f"lag_{i}" for i in range(1, N_LAGS + 1)]]
y = df["gdp_growth"]

# ── AR POOS ───────────────────────────────────────────────────────────────────
print("\n=== Autoregressive Model ===")
_, ar_out, ar_rmse, ar_mae = poos.poos_validation(
    method=autoregressive.ar_model_nowcast,
    X=X,
    y=y,
    prop_train=PROP_TRAIN,
)

# ── RF POOS ───────────────────────────────────────────────────────────────────
print("\n=== Random Forest Model ===")
_, rf_out, rf_rmse, rf_mae = poos.poos_validation(
    method=rf_benchmark.rf_model_nowcast,
    X=X,
    y=y,
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
poos.plot_poos_results(y, ar_out, title="Autoregressive Model POOS")
poos.plot_poos_results(y, rf_out, title="Random Forest Model POOS")
