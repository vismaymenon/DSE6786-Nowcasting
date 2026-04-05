from database.client import get_backend_client
from pipeline.output import res_main
from pipeline.load_data import load_main
from pipeline.fred_loader import sync_csv_to_supabase
from pipeline.ragged_edge import fill_ragged_edge, upsert_table
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

def historical_run(supabase, date):
    load_main(run_date=date)
    sync_csv_to_supabase(supabase)

    df_filled_md = fill_ragged_edge(supabase, "fred_md", freq="M")
    df_filled_md["sasdate"] = df_filled_md["sasdate"].dt.strftime("%Y-%m-%d")
    upsert_table(supabase, "filled_md", df_filled_md)

    df_filled_qd = fill_ragged_edge(supabase, "fred_qd_x", freq="Q")
    df_filled_qd["sasdate"] = df_filled_qd["sasdate"].dt.strftime("%Y-%m-%d")
    upsert_table(supabase, "filled_qd", df_filled_qd)

    res_main(date)

if __name__ == "__main__":
    supabase = get_backend_client()
    dates = pd.date_range(start="2025-07-31", end="2026-02-28", freq="ME")
    for date in dates:
        print(f"Running historical pipeline for {date.strftime('%Y-%m-%d')} ...")
        historical_run(supabase, date)