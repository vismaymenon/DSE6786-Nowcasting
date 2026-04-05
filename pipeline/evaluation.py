"""
evaluation.py
=============
Runs all nowcasting models via POOS and reports results.

Models
------
  1. AR benchmark        — AR(2) on GDP lags
  2. RF benchmark        — Random Forest on GDP lags
  3. LASSO               — LASSO on simple average (X1)
  4. LASSO lags          — LASSO on simple average + lags (X2)
  5. RF avg              — Random Forest on simple average + lags (X2)
  6. LASSO U-MIDAS       — LASSO on U-MIDAS (X3)
  7. RF U-MIDAS          — Random Forest on U-MIDAS (X3)
  8. Ensemble            — Average of models 3–7 (excluding benchmarks)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import pipeline.poos as poos
from pipeline.models.AR_benchmark import ar_model_nowcast
from pipeline.models.rf import randomForest
from pipeline.models.lasso import fit_lasso
from pipeline.output_x import (
    load_filled_data,
    build_X1, build_X2, build_X3, build_X4,
    build_X_AR, build_X_RF_bench,
)

NUM_TEST = 100


# =============================================================================
# STEP 1 — Load data and build feature matrices
# =============================================================================

print("=" * 60)
print("Step 1: Loading data and building feature matrices")
print("=" * 60)

df_md, df_qd = load_filled_data()

X_ar,       y_ar       = build_X_AR()
X_rf_bench, y_rf_bench = build_X_RF_bench()
X1,         y1         = build_X1(df_md, df_qd)
X2,         y2         = build_X2(df_md, df_qd)
X3,         y3         = build_X3(df_md, df_qd)
X4,         y4         = build_X4(df_md, df_qd)


# =============================================================================
# STEP 2 — Run POOS for each model
# =============================================================================

def run_poos(name, method, X, y):
    print(f"\n{'=' * 60}")
    print(f"Running POOS: {name}")
    print("=" * 60)
    _, y_df, rmse, mae = poos.poos_validation(method=method, X=X, y=y, num_test=NUM_TEST)
    return y_df, rmse, mae


ar_out,           ar_rmse,           ar_mae           = run_poos("AR Benchmark",    ar_model_nowcast, X_ar,       y_ar)
rf_bench_out,     rf_bench_rmse,     rf_bench_mae     = run_poos("RF Benchmark",    randomForest,     X_rf_bench, y_rf_bench)
lasso_out,        lasso_rmse,        lasso_mae        = run_poos("LASSO",           fit_lasso,        X1,         y1)
lasso_lags_out,   lasso_lags_rmse,   lasso_lags_mae   = run_poos("LASSO Simple Average lags",      fit_lasso,        X2,         y2)
rf_avg_out,       rf_avg_rmse,       rf_avg_mae       = run_poos("RF Simple Average",          randomForest,     X2,         y2)
lasso_umidas_out, lasso_umidas_rmse, lasso_umidas_mae = run_poos("LASSO U-MIDAS",   fit_lasso,        X3,         y3)
rf_umidas_out,    rf_umidas_rmse,    rf_umidas_mae    = run_poos("RF U-MIDAS",      randomForest,     X4,         y4)


# =============================================================================
# STEP 3 — Ensemble average (models 3–7, excluding benchmarks)
# =============================================================================

ensemble_dfs = [lasso_out, lasso_lags_out, rf_avg_out, lasso_umidas_out, rf_umidas_out]

# Align all to the same index (intersection)
common_idx = ensemble_dfs[0].index
for df in ensemble_dfs[1:]:
    common_idx = common_idx.intersection(df.index)

ensemble_y_hat       = np.mean([df.loc[common_idx, "y_hat"]         for df in ensemble_dfs], axis=0)
ensemble_50_lower    = np.mean([df.loc[common_idx, "pred_50_lower"]  for df in ensemble_dfs], axis=0)
ensemble_50_upper    = np.mean([df.loc[common_idx, "pred_50_upper"]  for df in ensemble_dfs], axis=0)
ensemble_80_lower    = np.mean([df.loc[common_idx, "pred_80_lower"]  for df in ensemble_dfs], axis=0)
ensemble_80_upper    = np.mean([df.loc[common_idx, "pred_80_upper"]  for df in ensemble_dfs], axis=0)
ensemble_y_true      = lasso_out.loc[common_idx, "y_true"]

ensemble_out = pd.DataFrame({
    "y_true":        ensemble_y_true.values,
    "y_hat":         ensemble_y_hat,
    "pred_50_lower": ensemble_50_lower,
    "pred_50_upper": ensemble_50_upper,
    "pred_80_lower": ensemble_80_lower,
    "pred_80_upper": ensemble_80_upper,
}, index=common_idx)

valid_ens = ensemble_out["y_true"].notna()
ensemble_rmse = float(np.sqrt(np.mean((ensemble_out.loc[valid_ens, "y_true"] - ensemble_out.loc[valid_ens, "y_hat"]) ** 2)))
ensemble_mae  = float(np.mean(np.abs(ensemble_out.loc[valid_ens, "y_true"] - ensemble_out.loc[valid_ens, "y_hat"])))


# =============================================================================
# STEP 4 — Results table
# =============================================================================

models = [
    ("AR Benchmark",   ar_rmse,           ar_mae,           ar_out),
    ("RF Benchmark",   rf_bench_rmse,     rf_bench_mae,     rf_bench_out),
    ("LASSO",          lasso_rmse,        lasso_mae,        lasso_out),
    ("LASSO lags",     lasso_lags_rmse,   lasso_lags_mae,   lasso_lags_out),
    ("RF avg",         rf_avg_rmse,       rf_avg_mae,       rf_avg_out),
    ("LASSO U-MIDAS",  lasso_umidas_rmse, lasso_umidas_mae, lasso_umidas_out),
    ("RF U-MIDAS",     rf_umidas_rmse,    rf_umidas_mae,    rf_umidas_out),
    ("Ensemble",       ensemble_rmse,     ensemble_mae,     ensemble_out),
]

print("\n" + "=" * 100)
print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8}  {'50CI Lower':>12} {'50CI Upper':>12} {'80CI Lower':>12} {'80CI Upper':>12}")
print("─" * 100)
for name, rmse, mae, out in models:
    last = out.iloc[-1]
    print(f"{name:<20} {rmse:>8.4f} {mae:>8.4f}  "
          f"{last['pred_50_lower']:>12.4f} {last['pred_50_upper']:>12.4f} "
          f"{last['pred_80_lower']:>12.4f} {last['pred_80_upper']:>12.4f}")
print("=" * 100)
print(f"\nOOS observations: {NUM_TEST}  |  CI shown for last OOS quarter: {out.index[-1].date()}")


# =============================================================================
# STEP 5 — Plots
# =============================================================================

poos.plot_poos_results(y_ar,       ar_out,           title="AR Benchmark — POOS")
poos.plot_poos_results(y_rf_bench, rf_bench_out,     title="RF Benchmark — POOS")
poos.plot_poos_results(y1,         lasso_out,        title="LASSO (simple avg) — POOS")
poos.plot_poos_results(y2,         lasso_lags_out,   title="LASSO (avg + lags) — POOS")
poos.plot_poos_results(y2,         rf_avg_out,       title="RF avg — POOS")
poos.plot_poos_results(y3,         lasso_umidas_out, title="LASSO U-MIDAS — POOS")
poos.plot_poos_results(y3,         rf_umidas_out,    title="RF U-MIDAS — POOS")
poos.plot_poos_results(y1,         ensemble_out,     title="Ensemble (avg of models 3–7) — POOS")


# =============================================================================
# STEP 6 — Push to Supabase
# =============================================================================

# def push_to_supabase(models, run_date=None):
#     """
#     Push POOS results to the model_forecasts table in Supabase.

#     Table schema (long format, one row per model × quarter):
#         run_date      DATE   — date this pipeline was run
#         model_name    TEXT   — model identifier
#         quarter_date  DATE   — the quarter being forecast (last day of quarter)
#         month_date    DATE   — same as quarter_date here (quarterly model)
#         nowcast       FLOAT  — point forecast
#         ci_50_lb      FLOAT  — 50% CI lower bound
#         ci_50_ub      FLOAT  — 50% CI upper bound
#         ci_80_lb      FLOAT  — 80% CI lower bound
#         ci_80_ub      FLOAT  — 80% CI upper bound
#     """
#     from database.client import get_backend_client

#     if run_date is None:
#         run_date = pd.Timestamp.today().date().isoformat()

#     client = get_backend_client()
#     records = []

#     for model_name, _, _, y_df in models:
#         for quarter_date, row in y_df.iterrows():
#             # Convert quarter start date (e.g. 2025-03-01) to quarter end date (e.g. 2025-03-31)
#             quarter_end = pd.Timestamp(quarter_date) + pd.offsets.QuarterEnd(0)
#             records.append({
#                 "run_date":     run_date,
#                 "model_name":   model_name,
#                 "quarter_date": quarter_end.date().isoformat(),
#                 "month_date":   quarter_end.date().isoformat(),
#                 "nowcast":      float(row["y_hat"]),
#                 "ci_50_lb":     float(row["pred_50_lower"]),
#                 "ci_50_ub":     float(row["pred_50_upper"]),
#                 "ci_80_lb":     float(row["pred_80_lower"]),
#                 "ci_80_ub":     float(row["pred_80_upper"]),
#             })

#     client.table("model_forecasts").upsert(
#         records, on_conflict="model_name,quarter_date,month_date"
#     ).execute()
#     print(f"Upserted {len(records)} rows into 'model_forecasts'.")


# push_to_supabase(models)