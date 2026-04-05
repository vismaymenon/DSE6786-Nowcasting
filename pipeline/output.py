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
from pipeline.output_x import build_X1, build_X2, build_X3, build_X4, load_filled_data
from database.client import get_backend_client
from ragged_edge import read_table

import pandas as pd
from pathlib import Path

from supabase import Client
from database.client import get_backend_client

NUM_TEST   = 100
AR_LAGS    = 2
RF_LAGS    = 4

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def build_lag_features(gdp, n_lags):
    df = gdp["GDPC1_t"].rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)
    
    lag_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    df = df.dropna(subset=lag_cols)  # only drop rows where lags are NaN, not gdp_growth
    
    X = df[lag_cols]
    y = df["gdp_growth"]
    return X, y


def run_models():
    df_md, df_qd = load_filled_data()

    X1, y1 = build_X1(df_md, df_qd)
    X2, y2 = build_X2(df_md, df_qd, n_lags=4)
    X3, y3 = build_X3(df_md, df_qd)
    X4, y4 = build_X4(df_md, df_qd, n_monthly_lags=4, n_qd_lags=4)

    supabase = get_backend_client()
    gdp = read_table(supabase, "gdp")

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

    # # ── RF U-MIDAS POOS ───────────────────────────────────────────────────────────
    # print("\n=== RF U-MIDAS ===")
    # _, umidas_out, umidas_rmse, umidas_mae = poos.poos_validation(
    #     method=rf_umidas_module.rf_umidas_nowcast,
    #     X=X3,
    #     y=y3,
    #     num_test=NUM_TEST,
    # )

    # # ── LASSO POOS ───────────────────────────────────────────────────────────
    # print("\n=== LASSO ===")
    # # df_md_filled, df_qd_filled = rf_umidas_module.load_filled_data()
    # _, lasso_out, lasso_rmse, lasso_mae = poos.poos_validation(
    #     method=lasso_module.fit_lasso,
    #     X=X1,
    #     y=y1,
    #     num_test=NUM_TEST,
    # )

    # print("\n=== LASSO with umidas ===")
    # _, lasso_umidas_out, lasso_umidas_rmse, lasso_umidas_mae = poos.poos_validation(
    #     method=lasso_module.fit_lasso,
    #     X=X3,
    #     y=y3,
    #     num_test=NUM_TEST,
    # )


    # # ── RF Average POOS ───────────────────────────────────────────────────────────
    # print("\n=== RF Average (monthly mean aggregation) ===")
    # _, avg_out, avg_rmse, avg_mae = poos.poos_validation(
    #     method=rf_avg_module.rf_aggre_nowcast,
    #     X=X1,
    #     y=y1,
    #     num_test=NUM_TEST,
    # )

    
    return ar_out, rf_out#, avg_out, lasso_umidas_out

# # ── Results summary ───────────────────────────────────────────────────────────
# print("\n" + "=" * 55)
# print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8}")
# print("─" * 55)
# print(f"{'Autoregressive AR(2)':<25} {ar_rmse:>8.4f} {ar_mae:>8.4f}")
# print(f"{'RF Benchmark':<25} {rf_rmse:>8.4f} {rf_mae:>8.4f}")
# print(f"{'RF Average':<25} {avg_rmse:>8.4f} {avg_mae:>8.4f}")
# print(f"{'RF U-MIDAS':<25} {umidas_rmse:>8.4f} {umidas_mae:>8.4f}")
# print(f"{'LASSO':<25} {lasso_rmse:>8.4f} {lasso_mae:>8.4f}")
# print("=" * 55)
# print(f"\nOOS observations: {NUM_TEST}")

# # ── Confidence intervals (last row of each POOS output) ───────────────────────
# print("\n=== 50% and 80% CI — last OOS observation ===")
# for name, out in [("AR(2)", ar_out), ("RF Benchmark", rf_out), ("RF Average", avg_out), ("RF U-MIDAS", umidas_out) 
#                 ,("LASSO", lasso_out)
#                 ]:
#     row = out.iloc[-1]
#     print(f"\n{name}")
#     print(f"  Actual       : {row['y_true']:.4f}")
#     print(f"  Predicted    : {row['y_hat']:.4f}")
#     print(f"  50% CI       : [{row['pred_50_lower']:.4f}, {row['pred_50_upper']:.4f}]")
#     print(f"  80% CI       : [{row['pred_80_lower']:.4f}, {row['pred_80_upper']:.4f}]")

# # ── Plots ─────────────────────────────────────────────────────────────────────
# poos.plot_poos_results(y_ar,     ar_out,     title="Autoregressive AR(2) — POOS")
# poos.plot_poos_results(y_rf,     rf_out,     title="RF Benchmark — POOS")
# poos.plot_poos_results(y_avg,    avg_out,    title="RF Average — POOS")
# poos.plot_poos_results(y_umidas, umidas_out, title="RF U-MIDAS — POOS")
# poos.plot_poos_results(y_lasso, lasso_out,  title="LASSO — POOS")


# Push Data to Supabase
def push_results_to_supabase(client, models: dict, run_date=None):
    run_date = run_date or pd.Timestamp.today().date()
    last_day_of_month = (pd.Timestamp(run_date) + pd.offsets.MonthEnd(0)).date()
    if run_date != last_day_of_month:
        run_date = (pd.Timestamp(run_date) - pd.offsets.MonthEnd(1)).date()

    for model_name, poos_out in models:
        # Only push nowcast rows (where y_true is NA)
        nowcast_rows = poos_out[poos_out["y_true"].isna()]

        if nowcast_rows.empty:
            print(f"No nowcast rows found for model '{model_name}', skipping.")
            continue

        records = []
        for idx, row in nowcast_rows.iterrows():
            idx = pd.to_datetime(idx)  # ensure idx is Timestamp
            records.append({
                "run_date": run_date.strftime("%Y-%m-%d"),
                "model_name": model_name,
                "quarter_date": idx.to_period("Q").asfreq("M", how="end").to_timestamp().strftime("%Y-%m-%d"),
                "month_date": idx.to_period("M").to_timestamp().strftime("%Y-%m-%d"),
                "nowcast": row["y_hat"],
                "ci_50_lb": row["pred_50_lower"],
                "ci_50_ub": row["pred_50_upper"],
                "ci_80_lb": row["pred_80_lower"],
                "ci_80_ub": row["pred_80_upper"],
            })

        client.table("model_forecasts").upsert(records, on_conflict="model_name,quarter_date,month_date").execute()
        print(f"Upserted {len(records)} nowcast row(s) for model '{model_name}' into 'model_forecasts'.")

def main():
    client = get_backend_client()

    ar_out, rf_out = run_models()
    models = [("AR_Benchmark", ar_out), 
              ("RF_Benchmark", rf_out)]
            #  ("RF_Average", avg_out), 
            #  ("RF_UMIDAS", umidas_out), 
            #  ("LASSO_UMIDAS", lasso_umidas_out),
            #   ("LASSO_Lags_UMIDAS", lasso_out),
            #   ("LASSO_Lags_Average", lasso_out),
            # ("LASSO_Average", lasso_out)] ## DONT FORGET TO UPDATE THIS WHEN MODEL IS FINALISED
    
    push_results_to_supabase(client, models)

if __name__ == "__main__":
    main()