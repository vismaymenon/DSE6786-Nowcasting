import pandas as pd
from database.client import get_backend_client
from supabase import Client

def push_evaluation_to_supabase(client: Client, models: dict, run_date=None):
    # Pull GDP actuals from Supabase
    gdp_response = client.table("gdp").select("sasdate, GDPC1_t").execute()
    gdp_df = pd.DataFrame(gdp_response.data)
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    gdp_df = gdp_df.set_index("sasdate")

    # Collect y_hat for each model
    model_preds = {}
    for model_name, poos_out in models:
        model_preds[model_name] = poos_out["y_hat"]

    # Combine all model predictions into a wide dataframe
    eval_df = pd.DataFrame(model_preds)
    eval_df.index = pd.to_datetime(eval_df.index)

    # Join GDP actuals
    eval_df = eval_df.join(gdp_df["GDPC1_t"], how="left") # join based on index (sasdate)
    eval_df = eval_df.rename(columns={"GDPC1_t": "gdp_actual"})

    # Build records
    records = []
    for idx, row in eval_df.iterrows():
        record = {
            "quarter_date": idx.to_period("Q").asfreq("M", how="end").to_timestamp().date(),
            "gdp_actual": row["gdp_actual"],
            "AR_Benchmark": row["AR_Benchmark"],
            "RF_Benchmark": row["RF_Benchmark"],
            "RF_Average": row["RF_Average"],
            "RF_UMIDAS": row["RF_UMIDAS"],
            "LASSO_UMIDAS": row["LASSO_UMIDAS"],
            "LASSO_Lags_UMIDAS": row["LASSO_Lags_UMIDAS"],
            "LASSO_Average": row["LASSO_Average"],
            "LASSO_Lags_Average": row["LASSO_Lags_Average"],
            "All_Model_Average": row["All_Model_Average"],
        }
        records.append(record)

    client.table("evaluation").upsert(records, on_conflict="quarter_date").execute()
    print(f"Upserted {len(records)} rows into 'evaluation'.")