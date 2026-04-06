import pipeline.poos as poos
from pipeline.models.rf import randomForest
from pipeline.models.lasso import fit_lasso
from pipeline.models.AR_benchmark import ar_model_nowcast
import pipeline.models.lasso as lasso_module

from pipeline.output_x import build_X1, build_X2, build_X3, build_X4, load_filled_data
from database.client import get_backend_client
from ragged_edge import read_table
import pandas as pd
import numpy as np
from pathlib import Path

from supabase import Client
from database.client import get_backend_client

def nowcast_single(model, X: pd.DataFrame, y: pd.Series, gdp: pd.DataFrame) -> pd.DataFrame:
    """
    Wraps a nowcast model to produce a single prediction for gdp.index[-2].
    """
    target_idx = gdp.index[-2]

    # Calculate train_size: years since 1960-09-01 minus 100
    train_size = len(y) - 100

    # Get integer position of target_idx in X
    target_pos = X.index.get_loc(target_idx)

    # Slice X and y: train_size rows before target + target row itself
    window_start = target_pos - train_size
    X_window = X.iloc[window_start : target_pos + 1]
    y_window = y.iloc[window_start : target_pos + 1]

    # Run model — last row of X_window is the test point
    _, y_train_actual, y_train_predicted, _, y_test_actual, y_test_predicted = model(X_window, y_window).values()
    std_error = np.std(y_train_actual - y_train_predicted)

    return pd.DataFrame(
        index=[target_idx],
        data={
            "quarter": pd.Period(target_idx, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date(),
            "y_true":        float(y_test_actual),
            "y_hat":         float(y_test_predicted),
            "pred_50_lower": float(y_test_predicted) - 0.674 * std_error,
            "pred_50_upper": float(y_test_predicted) + 0.674 * std_error,
            "pred_80_lower": float(y_test_predicted) - 1.282 * std_error,
            "pred_80_upper": float(y_test_predicted) + 1.282 * std_error,
        }
    )


def nowcast_single_latest(model, X: pd.DataFrame, y: pd.Series, gdp: pd.DataFrame, model_name: str, client) -> pd.DataFrame:
    """
    Wraps a nowcast model to produce a single prediction for gdp.index[-1].
    If gdp.index[-2] is NA, fetches the latest nowcast for that quarter from Supabase
    and uses it to fill in the missing value before building the training window.
    """
    target_idx = gdp.index[-1]
    prev_idx   = gdp.index[-2]

    train_size = len(y) - 100

    # Fill in gdp.index[-2] if it is NA
    y_filled = y.copy()
    if pd.isna(y.loc[prev_idx]):
        quarter_date = pd.Period(prev_idx, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date().isoformat()

        response = (
            client.table("model_forecasts")
            .select("nowcast, month_date, run_date")
            .eq("model_name", model_name)
            .eq("quarter_date", quarter_date)
            .order("run_date", desc=True)
            .order("month_date", desc=True)
            .limit(1)
            .execute()  # ← no .single() so 0 rows won't raise
        )

        if response.data:
            fetched_nowcast = float(response.data[0]["nowcast"])
            print(f"  Filling gdp.index[-2] ({prev_idx}) with Supabase nowcast: {fetched_nowcast:.4f} "
                  f"(month: {response.data[0]['month_date']}, run: {response.data[0]['run_date']})")
            y_filled.loc[prev_idx] = fetched_nowcast
        else:
            fallback = float(y.dropna().mean())
            print(f"  No forecast found for {prev_idx} in Supabase — falling back to mean: {fallback:.4f}")
            y_filled.loc[prev_idx] = fallback

    # Get integer position of target_idx in X
    target_pos = X.index.get_loc(target_idx)

    # Slice X and y: train_size rows before target + target row itself
    window_start = target_pos - train_size
    X_window = X.iloc[window_start : target_pos + 1]
    y_window = y_filled.iloc[window_start : target_pos + 1]

    # Run model — last row of X_window is the test point
    _, y_train_actual, y_train_predicted, _, y_test_actual, y_test_predicted = model(X_window, y_window).values()
    std_error = np.std(y_train_actual - y_train_predicted)

    return pd.DataFrame(
        index=[target_idx],
        data={
            "quarter":       pd.Period(target_idx, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date(),
            "y_true":        float(y_test_actual),
            "y_hat":         float(y_test_predicted),
            "pred_50_lower": float(y_test_predicted) - 0.674 * std_error,
            "pred_50_upper": float(y_test_predicted) + 0.674 * std_error,
            "pred_80_lower": float(y_test_predicted) - 1.282 * std_error,
            "pred_80_upper": float(y_test_predicted) + 1.282 * std_error,
        }
    )

def run_models():
    df_md, df_qd = load_filled_data()

    supabase = get_backend_client()
    raw = supabase.table("gdp").select("*").order("sasdate").execute()
    gdp = pd.DataFrame(raw.data).set_index("sasdate")["GDPC1_t"]
    print(gdp.tail(5))

    X1, y1 = build_X1(df_md, df_qd)
    X2, y2 = build_X2(df_md, df_qd, n_lags=4)
    X3, y3 = build_X3(df_md, df_qd)
    X4, y4 = build_X4(df_md, df_qd, n_monthly_lags=4, n_qd_lags=4)

    print(y1.tail(5))
    print(y2.tail(5))
    print(y3.tail(5))
    print(y4.tail(5))

    print("\n=== Nowcast for previous quarter ===")

    #ar_out = nowcast_single(autoregressive.run_ar_benchmark, X, y, gdp)
    #rf_out = nowcast_single(rf_benchmark.run_rf_benchmark, X, y, gdp)
    rf_avg_out = nowcast_single(rf_avg_module.rf_aggre_nowcast, X2, y2, gdp)
    rf_umidas_out = nowcast_single(rf_umidas_module.fit_rf_umidas, X4, y4, gdp)
    lasso_out = nowcast_single(lasso_module.fit_lasso, X1, y1, gdp)
    lasso_umidas_out = nowcast_single(lasso_module.fit_lasso, X3, y3, gdp)
    lasso_lags_out = nowcast_single(lasso_module.fit_lasso, X2, y2, gdp)
    lasso_umidas_lags_out = nowcast_single(lasso_module.fit_lasso, X4, y4, gdp)

    print("\n=== Nowcast for latest quarter (with gdp.index[-2] filled from Supabase if needed) ===")

    #ar_out = nowcast_single_latest(autoregressive.run_ar_benchmark, X, y, gdp)
    #rf_out = nowcast_single_latest(rf_benchmark.run_rf_benchmark, X, y, gdp)
    rf_avg_out_latest = nowcast_single_latest(randomForest, X2, y2, gdp, model_name = "RF_Average", client=supabase)
    rf_umidas_out_latest = nowcast_single_latest(randomForest, X4, y4, gdp, model_name = "RF_UMIDAS", client=supabase)
    lasso_out_latest = nowcast_single_latest(fit_lasso, X1, y1, gdp, model_name = "LASSO_Average", client=supabase)
    lasso_umidas_out_latest = nowcast_single_latest(fit_lasso, X3, y3, gdp, model_name = "LASSO_UMIDAS", client=supabase)
    lasso_lags_out_latest = nowcast_single_latest(fit_lasso, X2, y2, gdp, model_name = "LASSO_Lags_Average", client=supabase)
    lasso_umidas_lags_out_latest = nowcast_single_latest(fit_lasso, X4, y4, gdp, model_name = "LASSO_Lags_UMIDAS", client=supabase)

    models =[("RF_Average", rf_avg_out, rf_avg_out_latest), 
             ("RF_UMIDAS", rf_umidas_out, rf_umidas_out_latest), 
             ("LASSO_UMIDAS", lasso_umidas_out, lasso_umidas_out_latest),
             ("LASSO_Lags_UMIDAS", lasso_umidas_lags_out, lasso_umidas_lags_out_latest),
             ("LASSO_Lags_Average", lasso_lags_out, lasso_lags_out_latest),
             ("LASSO_Average", lasso_out, lasso_out_latest),
             ]
    
    push_results_to_supabase(supabase, models)

def push_results_to_supabase(client, models: list, run_date=None):
    run_date = pd.Timestamp(run_date or pd.Timestamp.today()).date()
    last_day_of_month = (pd.Timestamp(run_date) + pd.offsets.MonthEnd(0)).date()
    if run_date != last_day_of_month:
        run_date = (pd.Timestamp(run_date) - pd.offsets.MonthEnd(1)).date()

    month_date = run_date.strftime("%Y-%m-%d")

    for model_name, out, out_latest in models:
        records = []

        for df in [out, out_latest]:
            for idx, row in df.iterrows():
                idx = pd.to_datetime(idx)
                records.append({
                    "run_date":     run_date.strftime("%Y-%m-%d"),
                    "model_name":   model_name,
                    "quarter_date": idx.to_period("Q").asfreq("M", how="end").to_timestamp().strftime("%Y-%m-%d"),
                    "month_date":   month_date,
                    "nowcast":      row["y_hat"],
                    "ci_50_lb":     row["pred_50_lower"],
                    "ci_50_ub":     row["pred_50_upper"],
                    "ci_80_lb":     row["pred_80_lower"],
                    "ci_80_ub":     row["pred_80_upper"],
                })

        if not records:
            print(f"No records to push for model '{model_name}', skipping.")
            continue

        seen = set()
        unique_records = []
        for r in records:
            key = (r["model_name"], r["quarter_date"], r["month_date"])
            if key not in seen:
                seen.add(key)
                unique_records.append(r)

        client.table("model_forecasts").upsert(unique_records, on_conflict="model_name,quarter_date,month_date").execute()
        print(f"Upserted {len(unique_records)} record(s) for model '{model_name}' into 'model_forecasts'.")