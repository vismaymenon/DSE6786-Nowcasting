import pandas as pd
import numpy as np
from fredapi import Fred

def load_series_latest_release(series_id, api_key):
    """
    Fetches a specific data series and auto-labels it.

    Args:
        series_id (str): The unique FRED mnemonic (e.g., 'GDP', 'UNRATE').
        api_key (str): Your 32-character FRED API key.

    Returns:
        pd.Series: A pandas Series with a datetime index, where the 
                   '.name' attribute is set to the descriptive title.
    """    
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id)
    info = fred.get_series_info(series_id)
    data.name = info['title'] + ', ' + info['units']
    return data

def get_fred_md_metadata():
    """
    Scrapes the FRED-MD Monthly Database to map Series IDs to Transformation Codes.

    Args:
        None: Pulls directly from the St. Louis Fed's static data URL.

    Returns:
        dict: A dictionary where:
              - Key (str): The Series ID (e.g., 'RPI')
              - Value (int): The Transformation Code (1 through 7) 
                representing the math recommended by FRED for stationarity.
    """
    
    # Official URL for the current monthly FRED-MD file
    url = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2026-02-md.csv"
    
    # Read only the first two rows to get Column Names and Transformation Codes
    # Row 0: Column Names (RPI, INDPRO, etc.)
    # Row 1: Transformation Codes (5, 5, 2, etc.)
    metadata_df = pd.read_csv(url, nrows=1)
    
    # Create the dictionary: { 'Series_ID': Tcode }
    tcode_dict = metadata_df.iloc[0, 1:].to_dict()
    
    return tcode_dict


def transform_series(series, series_id, tcode_dict):
    """
    Transforms a pandas Series based on the FRED-MD Tcode.
    
    Parameters:
    series (pd.Series): The raw data from FRED.
    tcode (int): The transformation code (1-7).
    
    Returns:
    pd.Series: The transformed (stationary) data.
    """
    # Ensure series is numeric and drop NaNs for calculation
    series = pd.to_numeric(series, errors='coerce')
    tcode = tcode_dict.get(series_id)
    series.name = series.name + " Transformed"
    
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
    

def load_transformed_series_latest_release(series_id, api_key):
    """
    Fetches a specific transformed data series and auto-labels it.

    Args:
        series_id (str): The unique FRED mnemonic (e.g., 'GDP', 'UNRATE').
        api_key (str): Your 32-character FRED API key.

    Returns:
        pd.Series: A pandas Series with a datetime index, where the 
                   '.name' attribute is set to the descriptive title.
    """    
    md_metadata = get_fred_md_metadata()
    data = load_series_latest_release(series_id, api_key)
    return transform_series(data, series_id, md_metadata)
