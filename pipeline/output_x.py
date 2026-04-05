"""
output_x.py
===========
Prepares feature matrices (X, y) for all model specifications.

Four datasets
-------------
  X1 — Simple average
        Monthly variables averaged within each quarter, joined with quarterly data.
        Shape: (n_quarters, n_monthly + n_quarterly)

  X2 — Simple average + lags
        Same as X1 (call it qd1), then add 4 quarterly lags of every column in qd1.
        Shape: (n_quarters, (n_monthly + n_quarterly) × 5)

  X3 — U-MIDAS
        Monthly variables kept as 3 separate features per quarter (_m1/_m2/_m3),
        joined with quarterly data.
        Shape: (n_quarters, n_monthly×3 + n_quarterly)

  X4 — U-MIDAS + lags
        U-MIDAS monthly block (current quarter) + 4 quarterly lags of that block
        (= 12 monthly observations per variable), plus quarterly data + 4 lags.
        Shape: (n_quarters, n_monthly×3×5 + n_quarterly×5)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

THIS_DIR     = Path(__file__).resolve().parent   # pipeline/
PROJECT_DIR  = THIS_DIR.parent                   # project root

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_DIR))
from ragged_edge import read_table
from database.client import get_backend_client


# =============================================================================
# DATA LOADING
# =============================================================================

def load_filled_data():
    """Fetch filled_md and filled_qd from Supabase."""
    print("Loading filled data from Supabase …")
    supabase = get_backend_client()

    df_md = read_table(supabase, "filled_md")
    df_md["sasdate"] = pd.to_datetime(df_md["sasdate"])
    df_md = df_md.sort_values("sasdate").reset_index(drop=True)
    print(f"  filled_md : {df_md.shape}")

    df_qd = read_table(supabase, "filled_qd")
    df_qd["sasdate"] = pd.to_datetime(df_qd["sasdate"])
    df_qd = df_qd.sort_values("sasdate").reset_index(drop=True)
    print(f"  filled_qd : {df_qd.shape}\n")

    return df_md, df_qd


def _load_gdp():
    supabase = get_backend_client()
    gdp = read_table(supabase, "gdp")
    gdp["sasdate"] = pd.to_datetime(gdp["sasdate"])
    gdp = gdp.set_index("sasdate").sort_index()
    gdp = gdp[gdp.index.notna()]
    return gdp


# =============================================================================
# SHARED HELPERS
# =============================================================================

def _average_monthly_to_quarterly(df_md: pd.DataFrame) -> pd.DataFrame:
    """
    Average the 3 monthly observations within each quarter per variable.
    Excludes COVID flag columns (kept as quarterly-level features in df_qd).
    Returns DataFrame indexed by quarter label, columns suffixed with _md.
    """
    df = df_md.copy().sort_values("sasdate").reset_index(drop=True)
    feature_cols = [c for c in df.columns
                    if c not in ("sasdate", "covid_crash", "covid_recover")]
    df["qtr_label"] = (
        df["sasdate"].dt.to_period("Q").dt.start_time
        + pd.DateOffset(months=2)
    )
    df_agg = (
        df.groupby("qtr_label")[feature_cols]
        .mean()
        .rename_axis("sasdate")
        .add_suffix("_md")
    )
    return df_agg


def _umidas_monthly_to_quarterly(df_md: pd.DataFrame) -> pd.DataFrame:
    """
    Convert monthly data to quarterly U-MIDAS features.
    Each variable becomes 3 columns: _m1 (oldest), _m2, _m3 (most recent).
    Excludes COVID flag columns.
    Returns DataFrame indexed by quarter label.
    """
    df = df_md.copy().sort_values("sasdate").reset_index(drop=True)
    df["qtr_label"] = (
        df["sasdate"].dt.to_period("Q").dt.start_time
        + pd.DateOffset(months=2)
    )
    df["month_pos"] = df.groupby("qtr_label").cumcount() + 1
    feature_cols = [c for c in df.columns
                    if c not in ("sasdate", "qtr_label", "month_pos",
                                 "covid_crash", "covid_recover")]
    df_pivot = df.pivot(index="qtr_label", columns="month_pos", values=feature_cols)
    df_pivot.columns = [f"{col}_m{pos}" for col, pos in df_pivot.columns]
    df_pivot.index.name = "sasdate"
    return df_pivot


def _prep_qd(df_qd: pd.DataFrame) -> pd.DataFrame:
    df = df_qd.copy()
    df["sasdate"] = pd.to_datetime(df["sasdate"])
    return df.set_index("sasdate").sort_index()


def _add_lags(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """
    Add n_lags quarterly lags of every column in df.
    Lag-k columns are named {col}_lag{k}.
    The lag-0 (current) columns keep their original names.
    """
    lagged = [df]
    for k in range(1, n_lags + 1):
        shifted = df.shift(k).add_suffix(f"_lag{k}")
        lagged.append(shifted)
    return pd.concat(lagged, axis=1)


def _finalise(X: pd.DataFrame, gdp: pd.DataFrame) -> tuple:
    """
    Align y to X's index, drop NaN rows, return (X, y).

    Rows where X is complete but y is NaN are retained only if they fall
    after the last known GDP date — these are the nowcast rows (e.g. 2026 Q1).
    Rows before the GDP series begins are dropped.

    NOTE: y will be NaN for nowcast rows. When passing to poos_validation(),
    use only rows where y.notna() for evaluation, and use the last row of X
    separately for the actual nowcast prediction.
    """
    X = X.reindex(gdp.index)
    y = gdp["GDPC1_t"]
    valid = X.notna().all(axis=1) 
    if (~valid).sum() > 0:
        print(f"  Dropping {(~valid).sum()} rows with NaNs.")
    return X[valid], y[valid]



# =============================================================================
# X1 — SIMPLE AVERAGE
# =============================================================================

def build_X1(df_md: pd.DataFrame, df_qd: pd.DataFrame) -> tuple:
    """
    X1: averaged monthly features + quarterly features.

    Monthly variables are averaged across the 3 months within each quarter.
    """
    gdp   = _load_gdp()
    df_avg = _average_monthly_to_quarterly(df_md)
    df_q   = _prep_qd(df_qd)
    X = df_avg.join(df_q, how="inner")
    X, y = _finalise(X, gdp)
    X, y = X.iloc[4:], y.iloc[4:]  # drop first 4 quarters to match X2/X4 lag-trimmed window
    print(f"X1 (avg):            {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X2 — SIMPLE AVERAGE + LAGS
# =============================================================================

def build_X2(df_md: pd.DataFrame, df_qd: pd.DataFrame, n_lags: int = 4) -> tuple:
    """
    X2: averaged monthly + quarterly (call it qd1), then add n_lags quarterly
    lags of every column in qd1.

    Total features = (n_monthly_avg + n_quarterly) × (1 + n_lags)
    """
    gdp   = _load_gdp()
    df_avg = _average_monthly_to_quarterly(df_md)
    df_q   = _prep_qd(df_qd)
    qd1 = df_avg.join(df_q, how="inner")
    X = _add_lags(qd1, n_lags)
    X, y = _finalise(X, gdp)
    print(f"X2 (avg + {n_lags} lags):     {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X3 — U-MIDAS
# =============================================================================

def build_X3(df_md: pd.DataFrame, df_qd: pd.DataFrame) -> tuple:
    """
    X3: U-MIDAS monthly features (_m1/_m2/_m3) + quarterly features.

    Each monthly variable becomes 3 quarterly features preserving
    within-quarter dynamics.
    """
    gdp      = _load_gdp()
    df_umidas = _umidas_monthly_to_quarterly(df_md)
    df_q      = _prep_qd(df_qd)
    X = df_umidas.join(df_q, how="inner")
    X, y = _finalise(X, gdp)
    X, y = X.iloc[4:], y.iloc[4:]  # drop first 4 quarters to match X2/X4 lag-trimmed window
    print(f"X3 (U-MIDAS):        {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X4 — U-MIDAS + LAGS
# =============================================================================

def build_X4(df_md: pd.DataFrame, df_qd: pd.DataFrame,
             n_monthly_lags: int = 4, n_qd_lags: int = 4) -> tuple:
    """
    X4: U-MIDAS monthly block + n_monthly_lags quarterly lags of that block,
    plus quarterly data + n_qd_lags quarterly lags.

    n_monthly_lags=4 means 4 quarterly shifts of the _m1/_m2/_m3 block,
    covering 4×3=12 monthly observations of history per variable.

    n_qd_lags=4 means 4 quarterly lags of each quarterly variable.
    """
    gdp       = _load_gdp()
    df_umidas = _umidas_monthly_to_quarterly(df_md)
    df_q      = _prep_qd(df_qd)

    df_umidas_lagged = _add_lags(df_umidas, n_monthly_lags)
    df_q_lagged      = _add_lags(df_q, n_qd_lags)

    X = df_umidas_lagged.join(df_q_lagged, how="inner")
    X, y = _finalise(X, gdp)
    print(f"X4 (U-MIDAS + lags): {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X_AR — AUTOREGRESSIVE BENCHMARK (2 GDP lags)
# =============================================================================

def build_X_AR(n_lags: int = 2) -> tuple:
    """
    X_AR: 2 quarterly lags of GDP growth as features.
    Used by the AR benchmark model.
    """
    gdp = _load_gdp()
    df = gdp["GDPC1_t"].rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)
    lag_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    df = df[df[lag_cols].notna().all(axis=1)]  # only drop rows where lags are NaN
    X = df[lag_cols]
    y = df["gdp_growth"]
    X, y = X.iloc[2:], y.iloc[2:]
    print(f"X_AR ({n_lags} lags):          {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# X_RF_BENCH — BENCHMARK RF (4 GDP lags)
# =============================================================================

def build_X_RF_bench(n_lags: int = 4) -> tuple:
    """
    X_RF_bench: 4 quarterly lags of GDP growth as features.
    Used by the benchmark RF model.
    """
    gdp = _load_gdp()
    df = gdp["GDPC1_t"].rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)
    lag_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    df = df[df[lag_cols].notna().all(axis=1)]  # only drop rows where lags are NaN
    X = df[lag_cols]
    y = df["gdp_growth"]
    print(f"X_RF_bench ({n_lags} lags):    {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    df_md, df_qd = load_filled_data()

    X1, y1 = build_X1(df_md, df_qd)
    X2, y2 = build_X2(df_md, df_qd, n_lags=4)
    X3, y3 = build_X3(df_md, df_qd)
    X4, y4 = build_X4(df_md, df_qd, n_monthly_lags=4, n_qd_lags=4)
    X_AR, y_AR = build_X_AR(n_lags=2)
    X_RF_bench, y_RF_bench = build_X_RF_bench(n_lags=4)


    print("\n=== Summary ===")
    for name, X, y in [("X1 (avg)", X1, y1), ("X2 (avg + 4 lags)", X2, y2),
                        ("X3 (U-MIDAS)", X3, y3), ("X4 (U-MIDAS + lags)", X4, y4),
                        ("X_AR (AR benchmark)", X_AR, y_AR), ("X_RF_bench (RF benchmark)", X_RF_bench, y_RF_bench)]:
        print(f"\n--- {name} : {X.shape} ---")
        print("y:")
        print(y.head().to_string())
        print(y.tail().to_string())
