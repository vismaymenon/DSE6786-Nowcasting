import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import pipeline.poos as poos
import pipeline.models.autoregressive as autoregressive
import pipeline.models.rf_benchmark as rf_benchmark
import pipeline.models.rf_UMIDAS as rf_umidas_module
import pipeline.models.rf_avg as rf_avg_module
import pipeline.models.lasso as lasso_module

import pandas as pd
from pathlib import Path

NUM_TEST   = 100
AR_LAGS    = 2
RF_LAGS    = 4

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def build_lag_features(gdp, n_lags):
    df = gdp.rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)
    df.dropna(inplace=True)
    X = df[[f"lag_{i}" for i in range(1, n_lags + 1)]]
    y = df["gdp_growth"]
    return X, y


# ── Load data ─────────────────────────────────────────────────────────────────
gdp = pd.read_csv(DATA_DIR / "gdp.csv", parse_dates=["sasdate"])
gdp = gdp.set_index("sasdate").sort_index().squeeze()
gdp = gdp[gdp.index.notna()]

# ── AR(2) POOS ────────────────────────────────────────────────────────────────
print("\n=== Autoregressive Model AR(2) ===")
X_ar, y_ar = build_lag_features(gdp, AR_LAGS)
_, ar_out, ar_rmse, ar_mae = poos.poos_validation(
    method=autoregressive.ar_model_nowcast,
    X=X_ar,
    y=y_ar,
    num_test=NUM_TEST,
)

# ── RF Benchmark POOS ─────────────────────────────────────────────────────────
print("\n=== Random Forest Benchmark ===")
X_rf, y_rf = build_lag_features(gdp, RF_LAGS)
_, rf_out, rf_rmse, rf_mae = poos.poos_validation(
    method=rf_benchmark.rf_model_nowcast,
    X=X_rf,
    y=y_rf,
    num_test=NUM_TEST,
)

# ── RF U-MIDAS POOS ───────────────────────────────────────────────────────────
print("\n=== RF U-MIDAS ===")
df_md_filled, df_qd_filled = rf_umidas_module.load_filled_data()
X_umidas, y_umidas = rf_umidas_module.build_feature_matrix(df_md_filled, df_qd_filled)
_, umidas_out, umidas_rmse, umidas_mae = poos.poos_validation(
    method=rf_umidas_module.rf_umidas_nowcast,
    X=X_umidas,
    y=y_umidas,
    num_test=NUM_TEST,
)

# # ── RF LASSO POOS ───────────────────────────────────────────────────────────
# print("\n=== LASSO ===")
# df_md_filled, df_qd_filled = rf_umidas_module.load_filled_data()
# X_lasso= lasso_module.monthly_to_quarterly(df_md_filled)
# y_lasso = gdp
# _, lasso_out, lasso_rmse, lasso_mae = poos.poos_validation(
#     method=lasso_module.fit_lasso,
#     X=X_lasso,
#     y=y_lasso,
#     num_test=NUM_TEST,
# )


# ── RF Average POOS ───────────────────────────────────────────────────────────
print("\n=== RF Average (monthly mean aggregation) ===")
X_avg, y_avg = rf_avg_module.build_feature_matrix(df_md_filled, df_qd_filled)
_, avg_out, avg_rmse, avg_mae = poos.poos_validation(
    method=rf_avg_module.rf_aggre_nowcast,
    X=X_avg,
    y=y_avg,
    num_test=NUM_TEST,
)

# ── Results summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8}")
print("─" * 55)
print(f"{'Autoregressive AR(2)':<25} {ar_rmse:>8.4f} {ar_mae:>8.4f}")
print(f"{'RF Benchmark':<25} {rf_rmse:>8.4f} {rf_mae:>8.4f}")
print(f"{'RF Average':<25} {avg_rmse:>8.4f} {avg_mae:>8.4f}")
print(f"{'RF U-MIDAS':<25} {umidas_rmse:>8.4f} {umidas_mae:>8.4f}")
# print(f"{'LASSO':<25} {lasso_rmse:>8.4f} {lasso_mae:>8.4f}")
print("=" * 55)
print(f"\nOOS observations: {NUM_TEST}")

# ── Confidence intervals (last row of each POOS output) ───────────────────────
print("\n=== 50% and 80% CI — last OOS observation ===")
for name, out in [("AR(2)", ar_out), ("RF Benchmark", rf_out), ("RF Average", avg_out), ("RF U-MIDAS", umidas_out) 
                #   ,("LASSO", lasso_out)
                  ]:
    row = out.iloc[-1]
    print(f"\n{name}")
    print(f"  Actual       : {row['y_true']:.4f}")
    print(f"  Predicted    : {row['y_hat']:.4f}")
    print(f"  50% CI       : [{row['pred_50_lower']:.4f}, {row['pred_50_upper']:.4f}]")
    print(f"  80% CI       : [{row['pred_80_lower']:.4f}, {row['pred_80_upper']:.4f}]")

# ── Plots ─────────────────────────────────────────────────────────────────────
poos.plot_poos_results(y_ar,     ar_out,     title="Autoregressive AR(2) — POOS")
poos.plot_poos_results(y_rf,     rf_out,     title="RF Benchmark — POOS")
poos.plot_poos_results(y_avg,    avg_out,    title="RF Average — POOS")
poos.plot_poos_results(y_umidas, umidas_out, title="RF U-MIDAS — POOS")
# poos.plot_poos_results(y_umidas, lasso_out,  title="LASSO — POOS")
