import pandas as pd
from supabase import Client
from statsmodels.tsa.ar_model import AutoReg


def read_table(client: Client, table_name: str) -> pd.DataFrame:
    """Read an entire Supabase table into a DataFrame, handling pagination."""
    all_rows = []
    batch_size = 1000
    offset = 0

    while True:
        response = (
            client.table(table_name)
            .select("*")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        rows = response.data
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < batch_size:
            break
        offset += batch_size

    return pd.DataFrame(all_rows)


def upsert_table(client: Client, table_name: str, df: pd.DataFrame, on_conflict: str = "sasdate"):
    """Upsert a DataFrame into a Supabase table in batches."""
    records = df.where(pd.notnull(df), other=None).to_dict(orient="records")

    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        client.table(table_name).upsert(batch, on_conflict=on_conflict).execute()

    print(f"Upserted {len(records)} rows into '{table_name}'.")


def extend_time_index(df: pd.DataFrame, date_col: str, freq: str) -> pd.DataFrame:
    df = df.copy()

    if freq == "M":
        freq_pd = "MS"
    elif freq == "Q":
        freq_pd = "QS-MAR"
    else:
        raise ValueError("freq must be 'M' or 'Q'")

    target_date = (
        pd.Timestamp.today()
        .to_period("Q")
        .to_timestamp(how="end")
        .to_period("M")
        .to_timestamp()
    )

    if df[date_col].max() >= target_date:
        return df

    full_range = pd.date_range(start=df[date_col].min(), end=target_date, freq=freq_pd)
    df = df.set_index(date_col).reindex(full_range).reset_index()
    df = df.rename(columns={"index": date_col})

    return df

def _fill_series(series: pd.Series, p: int) -> pd.Series:
    series_filled = series.copy()

    first_obs_idx = series.first_valid_index()
    last_obs_idx = series.last_valid_index()

    if last_obs_idx is None:
        return series

    interpolated = series.loc[first_obs_idx:last_obs_idx].interpolate()
    series_filled.loc[first_obs_idx:last_obs_idx] = interpolated

    n_missing = len(series) - (last_obs_idx + 1)
    if n_missing <= 0:
        return series_filled

    train = interpolated.reset_index(drop=True)
    if len(train) <= p:
        return series_filled

    model = AutoReg(train, lags=p, old_names=False).fit()
    forecast = model.predict(start=len(train), end=len(train) + n_missing - 1)
    series_filled.iloc[last_obs_idx + 1 : last_obs_idx + 1 + n_missing] = forecast.values

    return series_filled


def fill_ragged_edge(client: Client, data_table: str, freq: str) -> pd.DataFrame:
    date_col = "sasdate"
    lag_csv = "data/bic_lags.csv"

    print(f"Reading '{data_table}'...")
    df = read_table(client, data_table)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)
    df = extend_time_index(df, date_col, freq)
    print(f"  -> {len(df)} rows after extending to {df[date_col].max().date()}")

    print(f"Reading '{lag_csv}'...")

    lag_df = pd.read_csv(lag_csv)
    lag_dict = dict(zip(lag_df["variable"], lag_df["lag"]))
    print(f"  -> {len(lag_dict)} variables to fill")

    df_filled = df.copy()
    for col in lag_dict:
        if col in df_filled.columns:
            df_filled[col] = pd.to_numeric(df_filled[col], errors="coerce")

    for i, (var, p) in enumerate(lag_dict.items(), 1):
        if var not in df_filled.columns:
            print(f"  [{i}/{len(lag_dict)}] '{var}' skipped (not in DataFrame)")
            continue
        print(f"  [{i}/{len(lag_dict)}] Filling '{var}' with AR({p})")
        df_filled[var] = _fill_series(df_filled[var], p)

    df_filled = df_filled.fillna(0)
    print(f"Done. Final shape: {df_filled.shape}")

    return df_filled