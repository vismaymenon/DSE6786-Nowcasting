import os
import pandas as pd
from pathlib import Path
from supabase import Client

DATA_DIR = Path(__file__).parent.parent / "data"
BATCH_SIZE = 500

CSV_FILES = {
    "gdp": DATA_DIR / "gdp.csv",
    "fred_md": DATA_DIR / "fred_md.csv",
    "fred_qd": DATA_DIR / "fred_qd.csv",
    "fred_qd_X": DATA_DIR / "fred_qd_X.csv"
}

def read_csv(file_path: Path):
    df = pd.read_csv(file_path, parse_dates=['sasdate'])
    df["sasdate"] = df["sasdate"].dt.strftime("%Y-%m-%d")
    df = df.where(pd.notnull(df), None)

    return df.to_dict(orient='records')

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