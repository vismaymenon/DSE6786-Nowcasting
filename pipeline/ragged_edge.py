import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

def fill_ragged_edge(data_csv, lag_csv):
    """
    Fills the ragged edge of monthly variables using AR(p).
    
    Parameters:
    - data_csv: path to CSV file containing monthly data (columns: variables)
    - lag_csv: path to CSV file containing lag values (columns: 'variable', 'lag')
    - date_col: optional name of date column to sort data by
    
    Returns:
    - df_filled: pandas DataFrame with ragged edge filled
    """
    
    # Load data
    df = pd.read_csv(data_csv)
    
    date_col = "sasdate"
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Load lags
    lag_df = pd.read_csv(lag_csv)
    lag_dict = dict(zip(lag_df['variable'], lag_df['lag']))
    
    # Ensure numeric columns only
    df_numeric = df.copy()
    for col in df.columns:
        if col in lag_dict:
            df_numeric[col] = pd.to_numeric(df[col], errors='coerce')
    
    df_filled = df_numeric.copy()
    
    # Function to fill a single series
    def fill_series(series, p):
        series_filled = series.copy()
        
        first_obs_idx = series.first_valid_index()
        last_obs_idx = series.last_valid_index()
        
        if last_obs_idx is None:
            return series
        
        # number of trailing steps
        interpolated = (series.loc[first_obs_idx:last_obs_idx].interpolate())
        series_filled.loc[first_obs_idx:last_obs_idx] = interpolated

        n_missing = len(series) - (last_obs_idx + 1)
        if n_missing <= 0:
            return series_filled
        
        # clean training data
        train = interpolated.reset_index(drop=True)
        
        if len(train) <= p:
            return series
        
        model = AutoReg(train, lags=p, old_names=False).fit()
        
        forecast = model.predict(
            start=len(train),
            end=len(train) + n_missing - 1
        )
        
        start = last_obs_idx + 1
        end = start + n_missing
        
        series_filled.iloc[start:end] = forecast.values
        
        return series_filled
    
    # Fill ragged edge for all variables in lag_dict
    for var, p in lag_dict.items():
        if var in df_filled.columns:
            df_filled[var] = fill_series(df_filled[var], p)
    
    return df_filled


def main():
    df_filled_md = fill_ragged_edge("data/fred_md.csv", "data/bic_lags.csv")
    df_filled_md.to_csv("data/filled_md.csv", index=False)
    df_filled_qd = fill_ragged_edge("data/fred_qd_X.csv", "data/bic_lags.csv")
    df_filled_qd.to_csv("data/filled_qd.csv", index=False)

if __name__ == "__main__":
    main()