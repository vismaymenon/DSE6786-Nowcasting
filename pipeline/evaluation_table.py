import pandas as pd
from database.client import get_backend_client

MODEL_COLUMNS = [
    "AR_Benchmark",
    "RF_Benchmark",
    "RF_Lags_Average",
    "RF_Lags_UMIDAS",
    "LASSO_UMIDAS",
    "LASSO_Average",
    "LASSO_Lags_Average",
    "All_Model_Average",
]

def compute_and_push_rmse(client) -> None:
    """
    Reads evaluation table, computes RMSE per model per version,
    and upserts into rmse table.
    """
    response = (
        client.table("evaluation")
        .select("version, gdp_actual, " + ", ".join(MODEL_COLUMNS))
        .not_.is_("gdp_actual", "null")   # only rows where actual is known
        .execute()
    )

    if not response.data:
        raise ValueError("No data found in evaluation table.")

    df = pd.DataFrame(response.data)
    df["version"] = df["version"].astype(float)

    records = []
    for version, group in df.groupby("version"):
        for model in MODEL_COLUMNS:
            col = group[model].dropna()
            actual = group.loc[col.index, "gdp_actual"]

            if len(col) == 0:
                continue

            rmse = float(((col - actual) ** 2).mean() ** 0.5)
            records.append({
                "model":   model,
                "version": version,
                "rmse":    rmse,
            })

    if not records:
        raise ValueError("No RMSE values could be computed.")

    client.table("rmse").upsert(records, on_conflict="model,version").execute()
    print(f"Pushed {len(records)} RMSE records.")

    if __name__ == "__main__":
        compute_and_push_rmse(client)
       