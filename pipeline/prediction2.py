import pandas as pd
import numpy as np
from database.client import get_backend_client
from pipeline.models.AR_benchmark import ar_model_nowcast
from pipeline.models.rf import randomForest
from pipeline.models.lasso import fit_lasso
from pipeline.output_x import build_X1, build_X2, build_X3, build_X4, load_filled_data, build_X_AR, build_X_RF_bench

def assign_version_prev(run_date):
    return (run_date.month - 1) % 3 + 4

def assign_version_latest(run_date):
    return (run_date.month - 1) % 3 + 1

def nowcast_single(model, X: pd.DataFrame, y: pd.Series, gdp: pd.DataFrame, model_name: str, client) -> pd.DataFrame:
    """
    Wraps a nowcast model to produce a single prediction for gdp.index[-2].
    CI is calculated using RMSE fetched from the evaluation table.
    """
    target_idx = gdp.index[-2]
    train_size = 166

    run_date = pd.Timestamp(run_date or pd.Timestamp.today()).date()
    last_day_of_month = (pd.Timestamp(run_date) + pd.offsets.MonthEnd(0)).date()
    if run_date != last_day_of_month:
        run_date = (pd.Timestamp(run_date) - pd.offsets.MonthEnd(1)).date()

    version = assign_version_prev(run_date)

    # Fetch RMSE from evaluation table
    rmse_response = (
        client.table("rmse")
        .select(model_name)
        .eq("version", version)
        .limit(1)
        .execute()
    )

    if not rmse_response.data:
        raise ValueError(f"No RMSE found in evaluation table for model '{model_name}'.")

    rmse = float(rmse_response.data[0][model_name])

    # Get integer position of target_idx in X
    target_pos = X.index.get_loc(target_idx)

    # Slice X and y: train_size rows before target + target row itself
    window_start = target_pos - train_size
    X_window = X.iloc[window_start : target_pos + 1]
    y_window = y.iloc[window_start : target_pos + 1]

    # Run model — last row of X_window is the test point
    _, _, _, _, y_test_actual, y_test_predicted = model(X_window, y_window).values()

    return pd.DataFrame(
        index=[target_idx],
        data={
            "version":       version,
            "quarter":       pd.Period(target_idx, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date(),
            "y_true":        float(y_test_actual),
            "y_hat":         float(y_test_predicted),
            "pred_50_lower": float(y_test_predicted) - 0.674 * rmse,
            "pred_50_upper": float(y_test_predicted) + 0.674 * rmse,
            "pred_80_lower": float(y_test_predicted) - 1.282 * rmse,
            "pred_80_upper": float(y_test_predicted) + 1.282 * rmse,
        }
    )

def nowcast_single_latest(model, X: pd.DataFrame, y: pd.Series, gdp: pd.DataFrame, model_name: str, client) -> pd.DataFrame:
    """
    Wraps a nowcast model to produce a single prediction for gdp.index[-1].
    If gdp.index[-2] is NA, fetches the latest nowcast for that quarter from Supabase
    and uses it to fill in the missing value before building the training window.
    CI is calculated using RMSE fetched from the evaluation table.
    """
    target_idx = gdp.index[-1]
    prev_idx   = gdp.index[-2]
    train_size = 166

    version = assign_version_latest(pd.Timestamp.today().date())

    # Fetch RMSE from evaluation table
    rmse_response = (
        client.table("rmse")
        .select(model_name)
        .eq("version", version)
        .limit(1)
        .execute()
    )

    if not rmse_response.data:
        raise ValueError(f"No RMSE found in evaluation table for model '{model_name}'.")

    rmse = float(rmse_response.data[0][model_name])

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
    _, _, _, _, y_test_actual, y_test_predicted = model(X_window, y_window).values()

    return pd.DataFrame(
        index=[target_idx],
        data={
            "version":       version,
            "quarter":       pd.Period(target_idx, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date(),
            "y_true":        float(y_test_actual),
            "y_hat":         float(y_test_predicted),
            "pred_50_lower": float(y_test_predicted) - 0.674 * rmse,
            "pred_50_upper": float(y_test_predicted) + 0.674 * rmse,
            "pred_80_lower": float(y_test_predicted) - 1.282 * rmse,
            "pred_80_upper": float(y_test_predicted) + 1.282 * rmse,
        }
    )

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