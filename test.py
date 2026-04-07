# run_pipeline.py
from database.client import get_backend_client
from pipeline.ragged_edge import fill_ragged_edge, upsert_table
from pipeline.fred_loader import sync_csv_to_supabase, fill_missing_gdp_quarters
from pipeline.load_data import load_main
import pandas as pd
from pipeline.evaluation_table_hist import push_forecasts_to_evaluation, calculate_and_upsert_rmse, calculate_mean_rmse_by_model
from pipeline.ci_update import update_ci_columns
from pipeline.prediction import run_all_nowcasts, compute_and_push_model_average


def run(run_date = None):
    load_main(run_date=pd.to_datetime(run_date) if run_date else None)
    supabase = get_backend_client()
    sync_csv_to_supabase(supabase)
    df_filled_md = fill_ragged_edge(supabase, "fred_md", freq="M")
    df_filled_md["sasdate"] = df_filled_md["sasdate"].dt.strftime("%Y-%m-%d")
    upsert_table(supabase, "filled_md", df_filled_md)

    df_filled_qd = fill_ragged_edge(supabase, "fred_qd_x", freq="Q")
    df_filled_qd["sasdate"] = df_filled_qd["sasdate"].dt.strftime("%Y-%m-%d")
    upsert_table(supabase, "filled_qd", df_filled_qd)
    fill_missing_gdp_quarters(supabase)

    gdp_response = supabase.table("gdp").select("sasdate, GDPC1_t").execute()
    gdp_response = get_backend_client().table("gdp").select("sasdate, GDPC1_t").order("sasdate", desc=False).execute()
    gdp_df = pd.DataFrame(gdp_response.data)
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    gdp_df = gdp_df.set_index("sasdate")


    quarter_dates = [
        pd.Period(gdp_df.index[-2], freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date().isoformat(),
        pd.Period(gdp_df.index[-1], freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date().isoformat(),
    ]

    run_all_nowcasts(gdp_df, supabase, run_date = pd.to_datetime(run_date) if run_date else None)
    compute_and_push_model_average(supabase, quarter_dates)

    push_forecasts_to_evaluation(supabase)
    calculate_and_upsert_rmse(supabase)
    calculate_mean_rmse_by_model(supabase)
    update_ci_columns(supabase)
    

if __name__ == "__main__":
    run()