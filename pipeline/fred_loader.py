import os
import pandas as pd
from pathlib import Path
from supabase import Client

DATA_DIR = Path(__file__).parent.parent / "data"
BATCH_SIZE = 500

CSV_FILES = {
    "gdp": DATA_DIR / "gdp.csv",
    "fred_md": DATA_DIR / "fred_md.csv",
    "fred_qd_x": DATA_DIR / "fred_qd_X.csv"
}

def read_csv(file_path: Path) -> list[dict]:
    df = pd.read_csv(file_path, parse_dates=['sasdate'])
    df = df.dropna(subset=["sasdate"])
    df["sasdate"] = df["sasdate"].dt.strftime("%Y-%m-%d")

    row = []
    for record in df.to_dict(orient='records'):
        cleaned = {
            k: (None if isinstance(v, float) and (pd.isna(v) or v == float("inf") or v == float("-inf")) else v)
            for k, v in record.items()
        }
        row.append(cleaned)
    return row

def upsert_table(
        client: Client,
        table_name: str,
        rows: list[dict]
) -> int:
    
    total = 0
    for i in range (0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        client.table(table_name).upsert(batch, on_conflict="sasdate").execute()

        total += len(batch)
        print(f"{table_name}: Upserted {total}/{len(rows)} rows")

    return total

def sync_csv_to_supabase(client:Client) -> None:
    
    print("\nSyncing CSV files to Supabase ...")
    for table_name, filepath in CSV_FILES.items():
        print(f"Reading {filepath.name} ...")
        if not filepath.exists():
            raise FileNotFoundError(
                f"Expected CSV file not found: {filepath} \n"
                f"Make sure load_data.main() has been run successfully first.")
        rows = read_csv(filepath)
        print(f"Upserting {len(rows)} rows to {table_name} ...")

        n = upsert_table(client, table_name, rows)
        print(f"Finished upserting {n} rows to {table_name}.")
    print("All CSV files synced to Supabase.")

def fill_missing_gdp_quarters(Client: Client) -> None:
    # 1. Get today's quarter
    target_date = pd.Timestamp.today()
    last_day_of_month = target_date + pd.offsets.MonthEnd(0)
    if target_date != last_day_of_month:
        target_date = target_date - pd.offsets.MonthEnd(1)
    

    current_period = pd.Period(target_date, freq='Q')

    # 2. Get the latest GDP entry
    response = Client.table('gdp').select('sasdate').order('sasdate', desc=True).limit(1).single().execute()

    if not response.data:
        raise Exception("No GDP entries found or error fetching data.")

    latest_period = pd.Period(response.data['sasdate'], freq='Q')

    # 3. Check if already up to date
    if latest_period >= current_period:
        print("GDP is already up to date.")
        return

    # 4. Fill in missing quarters
    missing_periods = pd.period_range(start=latest_period + 1, end=current_period, freq='Q')
    missing_rows = [{"sasdate": p.end_time.date().replace(day=1).isoformat()} for p in missing_periods]

    print(f"Inserting {len(missing_rows)} missing quarter(s): {missing_rows}")

    # 5. Upsert the missing rows
    Client.table('gdp').upsert(missing_rows, on_conflict='sasdate').execute()

    print("Successfully filled missing GDP quarters.")