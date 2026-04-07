from database.client import get_backend_client
import pandas as pd

def push_poos_to_supabase(client, models: list, run_date=None):
    run_date = pd.Timestamp(run_date or pd.Timestamp.today()).date()

    for model_name, poos_out, rmse, mae in models:
        records = []

        for idx, row in poos_out.iterrows():
            idx = pd.to_datetime(idx)
            records.append({
                "run_date":     run_date.strftime("%Y-%m-%d"),
                "model_name":   model_name,
                "quarter_date": idx.to_period("Q").asfreq("M", how="end").to_timestamp().strftime("%Y-%m-%d"),
                "month_date":   pd.Timestamp(row["month"]).strftime("%Y-%m-%d"),
                "nowcast":      None if pd.isna(row["y_hat"])         else float(row["y_hat"]),
                "ci_50_lb":     None if pd.isna(row["pred_50_lower"]) else float(row["pred_50_lower"]),
                "ci_50_ub":     None if pd.isna(row["pred_50_upper"]) else float(row["pred_50_upper"]),
                "ci_80_lb":     None if pd.isna(row["pred_80_lower"]) else float(row["pred_80_lower"]),
                "ci_80_ub":     None if pd.isna(row["pred_80_upper"]) else float(row["pred_80_upper"]),
            })

        if not records:
            print(f"No records to push for model '{model_name}', skipping.")
            continue

        client.table("model_forecasts").upsert(records, on_conflict="model_name,quarter_date,month_date").execute()
        print(f"Upserted {len(records)} POOS record(s) for model '{model_name}' into 'model_forecasts'.")

def push_evaluation_to_supabase(client, models: list, version: int, run_date=None):
    # Pull GDP actuals from Supabase
    gdp_response = client.table("gdp").select("sasdate, GDPC1_t").execute()
    gdp_df = pd.DataFrame(gdp_response.data)
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    gdp_df = gdp_df.set_index("sasdate")

    # Collect y_hat for each model
    model_preds = {}
    month_dates = None
    for model_name, poos_out, rmse, mae in models:
        model_preds[model_name] = poos_out["y_hat"]
        if month_dates is None:
            month_dates = poos_out["month"]  # use month from first model (same across all)

    # Combine all model predictions into a wide dataframe
    eval_df = pd.DataFrame(model_preds)
    eval_df.index = pd.to_datetime(eval_df.index)

    # Attach month_date
    eval_df["month_date"] = pd.to_datetime(month_dates)

    # Compute average across model columns only
    model_cols = list(model_preds.keys())
    eval_df["All_Model_Average"] = eval_df[model_cols].mean(axis=1)

    # Join GDP actuals
    eval_df = eval_df.join(gdp_df["GDPC1_t"], how="left")
    eval_df = eval_df.rename(columns={"GDPC1_t": "gdp_actual"})

    # Build records
    records = []
    for idx, row in eval_df.iterrows():
        record = {
            "quarter_date": pd.Timestamp(idx).to_period("Q").asfreq("M", how="end").to_timestamp().strftime("%Y-%m-%d"),
            "version":      version,
            "month_date":   pd.Timestamp(row["month_date"]).strftime("%Y-%m-%d"),
            "gdp_actual":   None if pd.isna(row["gdp_actual"]) else float(row["gdp_actual"]),
        }
        for col in model_cols + ["All_Model_Average"]:
            record[col] = None if pd.isna(row[col]) else float(row[col])
        records.append(record)

    client.table("evaluation").upsert(records, on_conflict="quarter_date,version").execute()
    print(f"Upserted {len(records)} rows into 'evaluation' (version={version}).")
    

def run():
    # POOS RESULTS TO BE ADDED HERE
    poos_version_4_to_6 = []

    push_poos_to_supabase(get_backend_client(), poos_version_4_to_6)

    # POOS EVALUATION TO BE ADDED HERE
    poos_version_1_to_3 = []
    push_poos_to_supabase(get_backend_client(), poos_version_1_to_3, version = 1)

if __name__ == "__main__":
    run()