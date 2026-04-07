from database.client import get_backend_client
from pipeline.load_data import load_main
from pipeline.fred_loader import sync_csv_to_supabase, fill_missing_gdp_quarters
from pipeline.ragged_edge import fill_ragged_edge, upsert_table
import warnings
warnings.filterwarnings("ignore")

from pipeline.models.rf import randomForest
from pipeline.models.lasso import fit_lasso
from pipeline.models.AR_benchmark import ar_model_nowcast

from pipeline.output_x import build_X1, build_X2, build_X3, build_X4, build_X_AR, build_X_RF_bench, load_filled_data
from database.client import get_backend_client
from prediction import nowcast_single, nowcast_single_latest, push_results_to_supabase
import pandas as pd
from supabase import Client
from database.client import get_backend_client

import pandas as pd

def historical_run(supabase, date):
    load_main(run_date=date)
    sync_csv_to_supabase(supabase)

    df_filled_md = fill_ragged_edge(supabase, "fred_md", freq="M", date=date)
    df_filled_md["sasdate"] = df_filled_md["sasdate"].dt.strftime("%Y-%m-%d")
    upsert_table(supabase, "filled_md", df_filled_md)

    df_filled_qd = fill_ragged_edge(supabase, "fred_qd_x", freq="Q", date=date)
    df_filled_qd["sasdate"] = df_filled_qd["sasdate"].dt.strftime("%Y-%m-%d")
    upsert_table(supabase, "filled_qd", df_filled_qd)

    fill_missing_gdp_quarters(supabase, date)

    # Load data
    df_md, df_qd = load_filled_data()
    X1, y1 = build_X1(df_md, df_qd)
    X2, y2 = build_X2(df_md, df_qd, n_lags=4)
    X3, y3 = build_X3(df_md, df_qd)
    X4, y4 = build_X4(df_md, df_qd, n_monthly_lags=4, n_qd_lags=4)
    X_ar, y_ar = build_X_AR()
    X_rf_bench, y_rf_bench = build_X_RF_bench()

    raw = supabase.table("gdp").select("*").order("sasdate").execute()
    gdp = pd.DataFrame(raw.data).set_index("sasdate")["GDPC1_t"]
    gdp.index = pd.to_datetime(gdp.index)

    # Edge case: if date is in Q3 2025 (Jul/Aug/Sep 2025), only run nowcast_single_latest
    date_period = pd.Period(date, freq="Q")
    q3_2025 = pd.Period("2025Q3", freq="Q")
    is_first_quarter = date_period == q3_2025

    if is_first_quarter:
        print(f"  Edge case: {date.strftime('%Y-%m-%d')} is in Q3 2025 — only running nowcast_single_latest.")
        models = [
            ("AR_Benchmark",        nowcast_single_latest(ar_model_nowcast, X_ar, y_ar, gdp, model_name="AR_Benchmark", client=supabase)),
            ("RF_Benchmark",        nowcast_single_latest(randomForest, X_rf_bench, y_rf_bench, gdp, model_name="RF_Benchmark", client=supabase)),
            ("RF_Average",          nowcast_single_latest(randomForest,   X2, y2, gdp, model_name="RF_Average",          client=supabase)),
            ("RF_UMIDAS",           nowcast_single_latest(randomForest,   X4, y4, gdp, model_name="RF_UMIDAS",           client=supabase)),
            ("LASSO_Average",       nowcast_single_latest(fit_lasso,           X1, y1, gdp, model_name="LASSO_Average",       client=supabase)),
            ("LASSO_UMIDAS",        nowcast_single_latest(fit_lasso,           X3, y3, gdp, model_name="LASSO_UMIDAS",        client=supabase)),
            ("LASSO_Lags_Average",  nowcast_single_latest(fit_lasso,           X2, y2, gdp, model_name="LASSO_Lags_Average",  client=supabase)),
            ("LASSO_Lags_UMIDAS",   nowcast_single_latest(fit_lasso,           X4, y4, gdp, model_name="LASSO_Lags_UMIDAS",   client=supabase)),
        ]
        # Wrap as (model_name, dummy_single, latest) to match push_results_to_supabase signature
        # Use latest as both to avoid pushing duplicates — push_results_to_supabase loops over [out, out_latest]
        # so we pass the same df twice and deduplicate via upsert
        models_full = [(name, out, out) for name, out in models]

    else:
        models_full = [
            ("AR_Benchmark",        nowcast_single(ar_model_nowcast, X_ar, y_ar, gdp),  nowcast_single_latest(ar_model_nowcast, X_ar, y_ar, gdp, model_name="AR_Benchmark", client=supabase)),
            ("RF_Benchmark",        nowcast_single(randomForest, X_rf_bench, y_rf_bench, gdp),  nowcast_single_latest(randomForest, X_rf_bench, y_rf_bench, gdp, model_name="RF_Benchmark", client=supabase))
            ("RF_Average",          nowcast_single(randomForest,  X2, y2, gdp),  nowcast_single_latest(randomForest,  X2, y2, gdp, model_name="RF_Average",         client=supabase)),
            ("RF_UMIDAS",           nowcast_single(randomForest,  X4, y4, gdp),  nowcast_single_latest(randomForest,  X4, y4, gdp, model_name="RF_UMIDAS",          client=supabase)),
            ("LASSO_Average",       nowcast_single(fit_lasso,          X1, y1, gdp),  nowcast_single_latest(fit_lasso,          X1, y1, gdp, model_name="LASSO_Average",      client=supabase)),
            ("LASSO_UMIDAS",        nowcast_single(fit_lasso,          X3, y3, gdp),  nowcast_single_latest(fit_lasso,          X3, y3, gdp, model_name="LASSO_UMIDAS",       client=supabase)),
            ("LASSO_Lags_Average",  nowcast_single(fit_lasso,          X2, y2, gdp),  nowcast_single_latest(fit_lasso,          X2, y2, gdp, model_name="LASSO_Lags_Average", client=supabase)),
            ("LASSO_Lags_UMIDAS",   nowcast_single(fit_lasso,          X4, y4, gdp),  nowcast_single_latest(fit_lasso,          X4, y4, gdp, model_name="LASSO_Lags_UMIDAS",   client=supabase)),
        ]

    push_results_to_supabase(supabase, models_full, run_date=date)


if __name__ == "__main__":
    supabase = get_backend_client()
    dates = pd.date_range(start="2025-07-31", end="2026-02-28", freq="ME")
    for date in dates:
        print(f"Running historical pipeline for {date.strftime('%Y-%m-%d')} ...")
        historical_run(supabase, date)