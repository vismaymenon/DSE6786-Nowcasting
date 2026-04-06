import pandas as pd
from database.client import get_backend_client
from supabase import Client

def push_evaluation_to_supabase(client, models: list, run_date=None):
    # Pull GDP actuals from Supabase
    gdp_response = client.table("gdp").select("sasdate, GDPC1_t").execute()
    gdp_df = pd.DataFrame(gdp_response.data)
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    gdp_df = gdp_df.set_index("sasdate")

    # Collect y_hat for each model
    model_preds = {}
    for model_name, poos_out, rmse, mae in models:
        model_preds[model_name] = poos_out["y_hat"]

    # Combine all model predictions into a wide dataframe
    eval_df = pd.DataFrame(model_preds)
    eval_df.index = pd.to_datetime(eval_df.index)

    # Compute average across all models
    eval_df["All_Model_Average"] = eval_df.mean(axis=1)

    # Join GDP actuals
    eval_df = eval_df.join(gdp_df["GDPC1_t"], how="left")
    eval_df = eval_df.rename(columns={"GDPC1_t": "gdp_actual"})

    # Build records
    records = []
    for idx, row in eval_df.iterrows():
        record = {"quarter_date": pd.Timestamp(idx).to_period("Q").asfreq("M", how="end").to_timestamp().strftime("%Y-%m-%d")}
        for col in eval_df.columns:
            value = row[col]
            record[col] = None if pd.isna(value) else float(value)
        records.append(record)

    client.table("evaluation").upsert(records, on_conflict="quarter_date").execute()
    print(f"Upserted {len(records)} rows into 'evaluation'.")