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

def nowcast_single(model, X: pd.DataFrame, y: pd.Series, gdp: pd.DataFrame, model_name: str, client, run_date = None) -> pd.DataFrame:
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

def nowcast_single_latest(model, X: pd.DataFrame, y: pd.Series, gdp: pd.DataFrame, model_name: str, client, run_date = None) -> pd.DataFrame:
    """
    Wraps a nowcast model to produce a single prediction for gdp.index[-1].
    If gdp.index[-2] is NA, fetches the latest nowcast for that quarter from Supabase
    and uses it to fill in the missing value before building the training window.
    CI is calculated using RMSE fetched from the evaluation table.
    """
    target_idx = gdp.index[-1]
    prev_idx   = gdp.index[-2]
    train_size = 166

    run_date = pd.Timestamp(run_date or pd.Timestamp.today()).date()
    last_day_of_month = (pd.Timestamp(run_date) + pd.offsets.MonthEnd(0)).date()
    if run_date != last_day_of_month:
        run_date = (pd.Timestamp(run_date) - pd.offsets.MonthEnd(1)).date()

    version = assign_version_latest(run_date)

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

X_ar , y_ar, X_rf, y_rf, X1, y1, X2, y2, X3, y3, X4, y4 = build_X_AR(), build_X_RF_bench(), build_X1(), build_X2(), build_X3(), build_X4()

MODEL_REGISTRY: dict[str, dict] = {
    "AR_Benchmark": {
        "model": ar_model_nowcast,
        "X": X_ar,
        "y": y_ar,
    },
    "RF_Benchmark": {
        "model": randomForest,
        "X": X_rf,
        "y": y_rf,
    },
    "RF_Lags_Average": {
        "model": randomForest,
        "X": X2,
        "y": y2,
    },
    "RF_Lags_UMIDAS": {
        "model": randomForest,
        "X": X4,
        "y": y4,
    },
    "LASSO_UMIDAS": {
        "model": fit_lasso,
        "X": X3,
        "y": y3,
    },
    "LASSO_Average": {
        "model": fit_lasso,
        "X": X1,
        "y": y1,
    },
    "LASSO_Lags_Average": {
        "model": fit_lasso,
        "X": X2,
        "y": y2,
    }
}


def _push_to_supabase(
    result: pd.DataFrame,
    model_name: str,
    run_date,
    client,
    *,
    push_evaluation: bool,
) -> None:
    """
    Pushes a single model result (one row) to model_forecasts.
    Optionally upserts into evaluation (only for nowcast_single, not latest).
    """
    row = result.iloc[0]
    quarter_date = str(row["quarter"])
    month_date   = str(pd.Timestamp(run_date).to_period("M").to_timestamp().date())

    # --- model_forecasts ---
    forecast_payload = {
        "run_date":     str(run_date),
        "model_name":   model_name,
        "quarter_date": quarter_date,
        "month_date":   month_date,
        "nowcast":      row["y_hat"],
        "ci_50_lb":     row["pred_50_lower"],
        "ci_50_ub":     row["pred_50_upper"],
        "ci_80_lb":     row["pred_80_lower"],
        "ci_80_ub":     row["pred_80_upper"],
    }
    client.table("model_forecasts").insert(forecast_payload).execute()

    # --- evaluation (one row per quarter×version, columns = model names) ---
    if push_evaluation:
        client.table("evaluation").upsert(
            {
                "quarter_date": quarter_date,
                "version":      int(row["version"]),
                "month_date":   month_date,
                "gdp_actual":   row["y_true"],
                model_name:     row["y_hat"],   # only this model's column
            },
            on_conflict="quarter_date,version",  # upsert merges columns
        ).execute()

def compute_and_push_model_average(client, quarter_dates: list[str], run_date=None) -> None:
    """
    Computes simple average across all models for the given quarter_dates only.
    """
    run_date = str(pd.Timestamp(run_date or pd.Timestamp.today()).date())

    response = (
        client.table("model_forecasts")
        .select("quarter_date, month_date, nowcast, ci_50_lb, ci_50_ub, ci_80_lb, ci_80_ub")
        .in_("quarter_date", quarter_dates)
        .neq("model_name", "All_Model_Average")
        .execute()
    )

    if not response.data:
        raise ValueError(f"No forecast data found for quarter_dates={quarter_dates}.")

    df = pd.DataFrame(response.data)
    numeric_cols = ["nowcast", "ci_50_lb", "ci_50_ub", "ci_80_lb", "ci_80_ub"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    records = []
    for (quarter_date, month_date), group in df.groupby(["quarter_date", "month_date"]):
        avg_row = group[numeric_cols].mean()
        records.append({
            "run_date":     run_date,
            "model_name":   "All_Model_Average",
            "quarter_date": quarter_date,
            "month_date":   month_date,
            "nowcast":      avg_row["nowcast"],
            "ci_50_lb":     avg_row["ci_50_lb"],
            "ci_50_ub":     avg_row["ci_50_ub"],
            "ci_80_lb":     avg_row["ci_80_lb"],
            "ci_80_ub":     avg_row["ci_80_ub"],
        })

    client.table("model_forecasts").upsert(
        records,
        on_conflict="model_name,quarter_date,month_date"
    ).execute()
    print(f"Pushed {len(records)} All_Model_Average records for {quarter_dates}.")

def run_all_nowcasts(
    gdp: pd.DataFrame,
    client,
    run_date=None,
) -> None:
    run_date = pd.Timestamp(run_date or pd.Timestamp.today()).date()

    # ── Step 1: historical nowcast (gdp.index[-2]) ──────────────────────────
    print("=== nowcast_single (prev quarter) ===")
    for model_name, cfg in MODEL_REGISTRY.items():
        print(f"  Running {model_name}...")
        result = nowcast_single(
            model=cfg["model"],
            X=cfg["X"], y=cfg["y"], gdp=gdp,
            model_name=model_name,
            client=client,
            run_date=run_date,
        )
        _push_to_supabase(result, model_name, run_date, client, push_evaluation=True)
        print(f"    → {result['y_hat'].iloc[0]:.4f}")

    # ── Step 2: latest nowcast (gdp.index[-1]) ───────────────────────────────
    print("\n=== nowcast_single_latest (current quarter) ===")
    for model_name, cfg in MODEL_REGISTRY.items():
        print(f"  Running {model_name}...")
        result = nowcast_single_latest(
            model=cfg["model"],
            X=cfg["X"], y=cfg["y"], gdp=gdp,
            model_name=model_name,
            client=client,
        )
        _push_to_supabase(result, model_name, run_date, client, push_evaluation=False)
        print(f"    → {result['y_hat'].iloc[0]:.4f}")

if __name__ == "__main__":
    supabase_client = get_backend_client()
    gdp_response = supabase_client.table("gdp").select("sasdate, GDPC1_t").execute()
    gdp_df = pd.DataFrame(gdp_response.data)
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    gdp_df = gdp_df.set_index("sasdate")

    quarter_dates = [
        pd.Period(gdp_df.index[-2], freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date().isoformat(),
        pd.Period(gdp_df.index[-1], freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date().isoformat(),
    ]

    run_all_nowcasts(gdp_df, supabase_client)
    compute_and_push_model_average(supabase_client, quarter_dates)
