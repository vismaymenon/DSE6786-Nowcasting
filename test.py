# run_pipeline.py
from database.client import get_backend_client
from pipeline.ragged_edge import fill_ragged_edge, upsert_table
from database.client import get_backend_client
from pipeline.fred_loader import sync_csv_to_supabase

def run():
    supabase = get_backend_client()
    sync_csv_to_supabase(supabase)
    df_filled_md = fill_ragged_edge(supabase, "fred_md", freq="M")
    df_filled_md["sasdate"] = df_filled_md["sasdate"].dt.strftime("%Y-%m-%d")
    upsert_table(supabase, "filled_md", df_filled_md)

    df_filled_qd = fill_ragged_edge(supabase, "fred_qd_x", freq="Q")
    df_filled_qd["sasdate"] = df_filled_qd["sasdate"].dt.strftime("%Y-%m-%d")
    upsert_table(supabase, "filled_qd", df_filled_qd)


if __name__ == "__main__":
    run()