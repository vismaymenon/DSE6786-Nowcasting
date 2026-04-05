import pandas as pd
import numpy as np
from fredapi import Fred

import os
from dotenv import load_dotenv

load_dotenv() 


# def load_series_latest_release(series_id, api_key):  
#     fred = Fred(api_key=api_key)
#     data = fred.get_series(series_id)
#     info = fred.get_series_info(series_id)
#     data.name = info['title'] + ', ' + info['units']
#     return data

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
    # Ensure series is numeric and drop NaNs for calculation
    series = pd.to_numeric(series, errors='coerce')
    tcode = tcode_dict.get(series_id)
    series.name = series.name + "_t"
    
    if tcode == 1: # No transformation
        return series
    
    elif tcode == 2: # First difference: x(t) - x(t-1)
        return series.diff()
    
    elif tcode == 3: # Second difference: (x(t) - x(t-1)) - (x(t-1) - x(t-2))
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
    

def load_transformed_series_latest_release(df, metadata):
    results = []   # ← separate list to collect transformed series
    bad_series = []

    for series_id in metadata.keys():
        try:
            print(f"Loading and transforming series: {series_id}")

            if series_id not in df.columns:
                raise ValueError(f"{series_id} not found in CSV")

            raw_series = df[series_id]
            raw_series.name = series_id
            transformed = transform_series(raw_series, series_id, metadata)
            results.append(transformed) 

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
    df = df.drop(columns=cols_to_drop)

    return df

def drop_empty_rows(df):
    # Drop rows where all values are NaN
    df = df.iloc[2:]  
    return df[df.index.notna()]

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

# print(load_transformed_series_latest_release(
#     load_series("https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2026-02-md.csv", skiprows=[1]),
#     get_fred_md_metadata(), 
#     API_KEY
#     ).head())

# print(load_transformed_series_latest_release(
#     load_series("https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/2026-02-qd.csv", skiprows=[1, 2]),
#     get_fred_md_metadata(), 
#     API_KEY
#     ).head())

def add_covid_flags(df):
    # add a flag column for those two quarters to indicate covid outliers
    df['covid_crash'] = 0
    df['covid_recover'] = 0
    df.loc['2020-04':'2020-06', 'covid_crash'] = 1
    df.loc['2020-07':'2020-09', 'covid_recover'] = 1
    return df

def load_main(run_date=None):
    if run_date is None:
        run_date = pd.Timestamp.today()
    
    last_day_of_month = run_date + pd.offsets.MonthEnd(0)
    if run_date != last_day_of_month:
        run_date = run_date - pd.offsets.MonthEnd(1)

    prev_month = (run_date - pd.DateOffset(months=1)).strftime("%Y-%m")

    try:
        fred_md = save_df(add_covid_flags(drop_empty_rows(load_transformed_series_latest_release(drop_columns(
            load_series(f"https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/{prev_month}-md.csv", skiprows=[1])),
            get_fred_md_metadata()
        ))), "../data", "fred_md")
        print(f"Finished loading and transforming monthly data for {prev_month}.")

        fred_qd = save_df(add_covid_flags(drop_empty_rows(load_transformed_series_latest_release(drop_columns(
            load_series(f"https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/{prev_month}-qd.csv", skiprows=[1, 2])),
            get_fred_qd_metadata()
        ))), "../data", "fred_qd")
        print(f"Finished loading and transforming quarterly data for {prev_month}.")

        fred_qd_X = save_df(fred_qd.iloc[:, 1:], "../data", "fred_qd_X")
        gdp = save_df(fred_qd.iloc[:, [0]]*400, "../data", "gdp")  
        
    except Exception as e:
        print(f"An error occurred during data loading and transformation: {e}")

if __name__ == "__main__":
    load_main(run_date=pd.Timestamp.today())
