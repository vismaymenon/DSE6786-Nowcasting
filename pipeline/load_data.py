import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FRED_API_KEY")


def get_fred_md_metadata():
    url = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"
    metadata_df = pd.read_csv(url, nrows=1)
    tcode_dict = metadata_df.iloc[0, 1:].to_dict()
    return tcode_dict

def get_fred_qd_metadata():
    url = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/current.csv"
    metadata_df = pd.read_csv(url, nrows=2)
    tcode_dict = metadata_df.iloc[1, 1:].to_dict()
    return tcode_dict

def load_series(url, skiprows=None):
    df = pd.read_csv(url, skiprows=skiprows)
    df["sasdate"] = pd.to_datetime(df["sasdate"])
    df = df.set_index("sasdate")
    return df

def transform_series(series, series_id, tcode_dict):
    series = pd.to_numeric(series, errors='coerce')
    tcode = tcode_dict.get(series_id)
    series.name = series.name + "_t"

    if tcode == 1:   # No transformation
        return series
    elif tcode == 2: # First difference: x(t) - x(t-1)
        return series.diff()
    elif tcode == 3: # Second difference: (x(t)-x(t-1)) - (x(t-1)-x(t-2))
        return series.diff().diff()
    elif tcode == 4: # Natural log: ln(x)
        return np.log(series)
    elif tcode == 5: # First difference of natural log: ln(x) - ln(x-1)
        return np.log(series).diff()
    elif tcode == 6: # Second difference of natural log
        return np.log(series).diff().diff()
    elif tcode == 7: # First difference of percent change
        return series.pct_change().diff()
    else:
        print(f"Unknown Tcode: {tcode}. Returning raw series.")
        return series

def load_transformed_series(df, metadata):
    results = []
    bad_series = []

    for series_id in metadata.keys():
        try:
            print(f"Loading and transforming series: {series_id}")

            if series_id not in df.columns:
                raise ValueError(f"{series_id} not found in CSV")

            raw_series = df[series_id]
            raw_series.name = series_id
            results.append(transform_series(raw_series, series_id, metadata))

        except Exception as e:
            bad_series.append(series_id)
            print(f"Error occurred while processing series {series_id}: {e}")

    print(f"\nFailed series: {bad_series}")
    return pd.concat(results, axis=1)

def drop_columns(df):
    # Drop column if it contains any NaN values in the first row to create a balanced panel
    nan_cols = df.columns[df.iloc[:1].isna().any()]

    # Drop irregular "OILPRICEx" column following McCracken and Ng (2016) recommendation
    cols_to_drop = list(nan_cols) + ["OILPRICEx"]
    return df.drop(columns=cols_to_drop)

def drop_empty_rows(df):
    return df.dropna(how='all')

def save_df(df, output_dir, file_name):
    if "__file__" in globals():
        base_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        base_dir = os.getcwd()
    resolved_dir = os.path.join(base_dir, output_dir)
    os.makedirs(resolved_dir, exist_ok=True)
    save_path = os.path.join(resolved_dir, f"{file_name}.csv")
    df.to_csv(save_path, header=True)
    print(f"  Saved to {save_path}")
    return df

def main():
    md_url = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"
    qd_url = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/current.csv"

    fred_md_raw = drop_columns(load_series(md_url, skiprows=[1]))
    fred_md = drop_empty_rows(load_transformed_series(fred_md_raw, get_fred_md_metadata()).bfill())
    save_df(fred_md, "../data", "fred_md")

    fred_qd_raw = drop_columns(load_series(qd_url, skiprows=[1, 2]))
    fred_qd = drop_empty_rows(load_transformed_series(fred_qd_raw, get_fred_qd_metadata()).bfill())
    save_df(fred_qd, "../data", "fred_qd")

    # Remove target variable from FRED QD
    save_df(drop_empty_rows(fred_qd.iloc[:, 1:]), "../data", "fred_qd_X")

    # Save GDP target variable separately, converted to annualized growth rate
    save_df(drop_empty_rows(fred_qd.iloc[:, 0] * 400), "../data", "gdp")

    print("Data loading and transformation complete.")

if __name__ == "__main__":
    main()
