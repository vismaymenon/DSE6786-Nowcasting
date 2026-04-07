from database.client import get_backend_client
import pandas as pd
import numpy as np
from pipeline.models.AR_benchmark import ar_model_nowcast
from pipeline.models.rf import randomForest
from pipeline.models.lasso import fit_lasso
from pipeline.output_x_poos import make_build_X, build_X1_from_cut, build_X2_from_cut, build_X3_from_cut, build_X4_from_cut, build_X_AR_from_cut, build_X_RF_bench_from_cut
from pipeline.poos import poos_validation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

def get_month_date(quarter_ts: pd.Timestamp, version: int) -> pd.Timestamp:
    """
    Given a quarter timestamp and version, return the last day of the target month.

    Version 1 = 1st month of same quarter      (e.g. Q1 → Jan 31)
    Version 2 = 2nd month of same quarter      (e.g. Q1 → Feb 28)
    Version 3 = 3rd month of same quarter      (e.g. Q1 → Mar 31)
    Version 4 = 1st month of next quarter      (e.g. Q1 → Apr 30)
    Version 5 = 2nd month of next quarter      (e.g. Q1 → May 31)
    Version 6 = 3rd month of next quarter      (e.g. Q1 → Jun 30)
    """
    if version not in range(1, 7):
        raise ValueError(f"version must be 1–6, got {version}")

    quarter_start = quarter_ts.to_period("Q").to_timestamp(how="start")  # e.g. 2024-01-01

    # Offset in months from quarter start: v1→0, v2→1, v3→2, v4→3, v5→4, v6→5
    month_offset = version - 1
    target = quarter_start + pd.DateOffset(months=month_offset)

    # Return last calendar day of that month
    return target + pd.offsets.MonthEnd(0)


def push_poos_to_supabase(client, models: list, version: int, run_date=None):
    run_date = pd.Timestamp(run_date or pd.Timestamp.today()).date()

    for model_name, poos_out, rmse, mae in models:
        records = []

        for idx, row in poos_out.iterrows():
            idx = pd.to_datetime(idx)
            quarter_date = (
                idx.to_period("Q").asfreq("M", how="end").to_timestamp()
            )
            month_date = get_month_date(idx, version)

            records.append({
                "run_date":     run_date.strftime("%Y-%m-%d"),
                "model_name":   model_name,
                "quarter_date": quarter_date.strftime("%Y-%m-%d"),
                "month_date":   month_date.strftime("%Y-%m-%d"),
                "nowcast":      None if pd.isna(row["y_hat"])         else float(row["y_hat"]),
                "ci_50_lb":     None if pd.isna(row["pred_50_lower"]) else float(row["pred_50_lower"]),
                "ci_50_ub":     None if pd.isna(row["pred_50_upper"]) else float(row["pred_50_upper"]),
                "ci_80_lb":     None if pd.isna(row["pred_80_lower"]) else float(row["pred_80_lower"]),
                "ci_80_ub":     None if pd.isna(row["pred_80_upper"]) else float(row["pred_80_upper"]),
            })

        if not records:
            print(f"No records to push for model '{model_name}', skipping.")
            continue

        client.table("model_forecasts").upsert(
            records, on_conflict="model_name,quarter_date,month_date"
        ).execute()
        print(f"Upserted {len(records)} POOS record(s) for model '{model_name}' into 'model_forecasts'.")


def push_evaluation_to_supabase(client, models: list, version: int, run_date=None):
    # Pull GDP actuals from Supabase
    gdp_response = client.table("gdp").select("sasdate, GDPC1_t").execute()
    gdp_df = pd.DataFrame(gdp_response.data)
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    gdp_df = gdp_df.set_index("sasdate")

    # Collect y_hat per model
    model_preds = {}
    for model_name, poos_out, rmse, mae in models:
        model_preds[model_name] = poos_out["y_hat"]

    if not model_preds:
        print("No models provided to push_evaluation_to_supabase, skipping.")
        return

    eval_df = pd.DataFrame(model_preds)
    eval_df.index = pd.to_datetime(eval_df.index)

    # Derive month_date from index + version (no longer relies on poos_out["month"])
    eval_df["month_date"] = eval_df.index.map(
        lambda ts: get_month_date(ts, version)
    )

    # Compute average across model columns only
    model_cols = list(model_preds.keys())
    eval_df["All_Model_Average"] = eval_df[model_cols].mean(axis=1)

    # Join GDP actuals — align on quarter-end month timestamp
    gdp_df.index = pd.to_datetime(gdp_df.index)
    eval_df = eval_df.join(gdp_df["GDPC1_t"], how="left")
    eval_df = eval_df.rename(columns={"GDPC1_t": "gdp_actual"})

    # Build records
    records = []
    for idx, row in eval_df.iterrows():
        quarter_date = (
            pd.Timestamp(idx).to_period("Q").asfreq("M", how="end").to_timestamp()
        )
        record = {
            "quarter_date": quarter_date.strftime("%Y-%m-%d"),
            "version":      version,
            "month_date":   pd.Timestamp(row["month_date"]).strftime("%Y-%m-%d"),
            "gdp_actual":   None if pd.isna(row["gdp_actual"]) else float(row["gdp_actual"]),
        }
        for col in model_cols + ["All_Model_Average"]:
            val = row[col]
            record[col] = None if pd.isna(val) else float(val)
        records.append(record)

    client.table("evaluation").upsert(
        records, on_conflict="quarter_date,version"
    ).execute()
    print(f"Upserted {len(records)} rows into 'evaluation' (version={version}).")


BUILD_REGISTRY = {
    # "AR_Benchmark":      "X_AR",
    # "RF_Lags_Average":   "X2",
    # "RF_Lags_UMIDAS":    "X4",
    # "LASSO_UMIDAS":      "X3",
    # "LASSO_Average":     "X1",
    "LASSO_Lags_Average": "X2",
}

MODEL_REGISTRY: dict[str, dict] = {
    # "AR_Benchmark": {
    #     "model": ar_model_nowcast
    # },
    # "RF_Lags_Average": {
    #     "model": randomForest
    # },
    # "RF_Lags_UMIDAS": {
    #     "model": randomForest
    # },
    # "LASSO_UMIDAS": {
    #     "model": fit_lasso
    # },
    # "LASSO_Average": {
    #     "model": fit_lasso

    # },
    "LASSO_Lags_Average": {
        "model": fit_lasso
    }
}

def plot_poos_results(
    y_full: pd.Series,
    y_df: pd.DataFrame,
    model_name: str,
    version: int,
    last_n: int = 200,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    y_plot = y_full.iloc[-last_n:]
    cutoff_date = y_plot.index[0]

    ax.plot(
        y_plot.index,
        y_plot.values,
        color="black",
        linewidth=1.2,
        label="Actual (full sample)",
        zorder=3,
    )

    y_df_plot = y_df[y_df.index >= cutoff_date]
    idx = y_df_plot.index

    ax.plot(
        idx,
        y_df_plot["y_hat"],
        color="red",
        linewidth=1.2,
        label="Predicted (OOS)",
        zorder=4,
    )

    ax.fill_between(
        idx,
        y_df_plot["pred_50_lower"],
        y_df_plot["pred_50_upper"],
        alpha=0.4,
        color="steelblue",
        label="50% CI",
    )

    ax.fill_between(
        idx,
        y_df_plot["pred_80_lower"],
        y_df_plot["pred_80_upper"],
        alpha=0.2,
        color="steelblue",
        label="80% CI",
    )

    ax.axvline(
        x=idx[0],
        color="grey",
        linestyle=":",
        linewidth=1,
        label="OOS start",
    )

    title = f"{model_name} — Version {version} — POOS Results"

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("GDP growth")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    os.makedirs("pipeline/plots", exist_ok=True)
    safe_title = title.replace(" ", "_").replace("/", "_")
    fig.savefig(
        os.path.join("pipeline/plots", f"{safe_title}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def run():
    client = get_backend_client()

    # ── Load data ─────────────────────────────────────────────────────────────
    QD_t = pd.read_csv("data/fred_qd_X.csv")
    QD_t["sasdate"] = pd.to_datetime(QD_t["sasdate"])

    MD_t = pd.read_csv("data/fred_md.csv")
    MD_t["sasdate"] = pd.to_datetime(MD_t["sasdate"])

    gdp_df = pd.read_csv("data/gdp.csv")
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    y_full = gdp_df.set_index("sasdate")["GDPC1_t"]

    # ── Run POOS per version and push ─────────────────────────────────────────
    for version in [4,5,6]:
        print(f"\n{'='*60}\nRunning POOS for version {version}\n{'='*60}")
        poos_results = []

        for model_name, cfg in MODEL_REGISTRY.items():
            build_name = BUILD_REGISTRY[model_name]
            print(f"\n--- {model_name} (buildX={build_name}) ---")

            poos_out, rmse, mae = poos_validation(
                method=cfg["model"],
                buildname=build_name,
                QD_t=QD_t,
                MD_t=MD_t,
                y_full=y_full,
                version=version,
            )

            plot_poos_results(y_full, poos_out, model_name, version)
            print(f"  {model_name} → RMSE={rmse:.4f} | MAE={mae:.4f}")
            poos_results.append((model_name, poos_out, rmse, mae))

        # Push forecasts for all versions
        push_poos_to_supabase(client, poos_results, version=version)

if __name__ == "__main__":
    run()