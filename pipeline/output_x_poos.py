import pandas as pd

def _average_monthly_to_quarterly_from_df(md_filled: pd.DataFrame) -> pd.DataFrame:
    """Averages monthly data to quarterly, indexed by last month of quarter."""
    md = md_filled.copy()
    md["sasdate"] = pd.to_datetime(md["sasdate"])
    md["quarter"] = (
        md["sasdate"]
        .dt.to_period("Q")
        .dt.to_timestamp(how="end")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    return md.drop(columns="sasdate").groupby("quarter").mean()


def _umidas_monthly_to_quarterly_from_df(md_filled: pd.DataFrame) -> pd.DataFrame:
    """
    U-MIDAS: expands each monthly variable into 3 columns (_m1, _m2, _m3)
    preserving within-quarter dynamics.
    """
    md = md_filled.copy()
    md["sasdate"] = pd.to_datetime(md["sasdate"])
    md["quarter"] = (
        md["sasdate"]
        .dt.to_period("Q")
        .dt.to_timestamp(how="end")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    md["month_in_q"] = md["sasdate"].dt.to_period("Q").apply(
        lambda p: md.loc[md["sasdate"].dt.to_period("Q") == p, "sasdate"].rank(method="first").astype(int)
    )
    # simpler: use position within quarter
    md["month_in_q"] = md.groupby("quarter")["sasdate"].rank(method="first").astype(int)

    feature_cols = [c for c in md.columns if c not in ("sasdate", "quarter", "month_in_q")]
    frames = []
    for m in [1, 2, 3]:
        slice_ = (
            md[md["month_in_q"] == m]
            .set_index("quarter")[feature_cols]
            .rename(columns={c: f"{c}_m{m}" for c in feature_cols})
        )
        frames.append(slice_)

    return pd.concat(frames, axis=1).sort_index()


def _prep_qd_from_df(qd_filled: pd.DataFrame) -> pd.DataFrame:
    """Sets sasdate as index for quarterly data."""
    qd = qd_filled.copy()
    qd["sasdate"] = pd.to_datetime(qd["sasdate"])
    return qd.set_index("sasdate")


def _build_gdp_lags_from_cut(gdp_cut: pd.Series, n_lags: int = 4) -> pd.DataFrame:
    """Builds GDP lag columns from the cut series."""
    gdp_lags = pd.DataFrame(index=gdp_cut.index)
    for lag in range(1, n_lags + 1):
        gdp_lags[f"gdp_lag_{lag}"] = gdp_cut.shift(lag)
    return gdp_lags


def _add_lags_df(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """Adds n quarterly lags of every column in df."""
    frames = [df]
    for lag in range(1, n_lags + 1):
        lagged = df.shift(lag).rename(columns={c: f"{c}_lag{lag}" for c in df.columns})
        frames.append(lagged)
    return pd.concat(frames, axis=1)


def _finalise_from_cut(X: pd.DataFrame, gdp_actual: pd.Series) -> tuple:
    """Aligns y to X index using actual GDP."""
    y = gdp_actual.reindex(X.index)
    return X, y


# ── X1: averaged monthly + quarterly + GDP lags ──────────────────────────────

def build_X1_from_cut(
    qd_filled: pd.DataFrame,
    md_filled: pd.DataFrame,
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_lags: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    df_avg     = _average_monthly_to_quarterly_from_df(md_filled)
    df_q       = _prep_qd_from_df(qd_filled)
    gdp_lags   = _build_gdp_lags_from_cut(gdp_cut, n_lags)
    X          = df_avg.join(df_q, how="inner").join(gdp_lags, how="left")
    X, y       = _finalise_from_cut(X, gdp_actual)
    print(f"X1 (avg):            {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# ── X2: averaged monthly + quarterly + lags of all columns + GDP lags ────────

def build_X2_from_cut(
    qd_filled: pd.DataFrame,
    md_filled: pd.DataFrame,
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_lags: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    df_avg     = _average_monthly_to_quarterly_from_df(md_filled)
    df_q       = _prep_qd_from_df(qd_filled)
    qd1        = df_avg.join(df_q, how="inner")
    gdp_lags   = _build_gdp_lags_from_cut(gdp_cut, n_lags)
    X          = _add_lags_df(qd1, n_lags).join(gdp_lags, how="left")
    X, y       = _finalise_from_cut(X, gdp_actual)
    print(f"X2 (avg + {n_lags} lags):     {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# ── X3: U-MIDAS monthly + quarterly + GDP lags ───────────────────────────────

def build_X3_from_cut(
    qd_filled: pd.DataFrame,
    md_filled: pd.DataFrame,
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_lags: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    df_umidas  = _umidas_monthly_to_quarterly_from_df(md_filled)
    df_q       = _prep_qd_from_df(qd_filled)
    gdp_lags   = _build_gdp_lags_from_cut(gdp_cut, n_lags)
    X          = df_umidas.join(df_q, how="inner").join(gdp_lags, how="left")
    X, y       = _finalise_from_cut(X, gdp_actual)
    print(f"X3 (U-MIDAS):        {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# ── X4: U-MIDAS + lags + quarterly + lags + GDP lags ────────────────────────

def build_X4_from_cut(
    qd_filled: pd.DataFrame,
    md_filled: pd.DataFrame,
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_monthly_lags: int = 4,
    n_qd_lags: int = 4,
    n_gdp_lags: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    df_umidas        = _umidas_monthly_to_quarterly_from_df(md_filled)
    df_q             = _prep_qd_from_df(qd_filled)
    df_umidas_lagged = _add_lags_df(df_umidas, n_monthly_lags)
    df_q_lagged      = _add_lags_df(df_q, n_qd_lags)
    gdp_lags         = _build_gdp_lags_from_cut(gdp_cut, n_gdp_lags)
    X                = df_umidas_lagged.join(df_q_lagged, how="inner").join(gdp_lags, how="left")
    X, y             = _finalise_from_cut(X, gdp_actual)
    print(f"X4 (U-MIDAS + lags): {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# ── X_AR: GDP lags only ───────────────────────────────────────────────────────

def build_X_AR_from_cut(
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_lags: int = 2,
) -> tuple[pd.DataFrame, pd.Series]:
    df = gdp_cut.rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)
    lag_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    df = df[df[lag_cols].notna().all(axis=1)]
    X  = df[lag_cols]
    y  = gdp_actual.reindex(X.index)
    print(f"X_AR ({n_lags} lags):          {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y


# ── X_RF_bench: GDP lags only (4 lags) ───────────────────────────────────────

def build_X_RF_bench_from_cut(
    gdp_cut: pd.Series,
    gdp_actual: pd.Series,
    n_lags: int = 4,
) -> tuple[pd.DataFrame, pd.Series]:
    df = gdp_cut.rename("gdp_growth").to_frame()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)
    lag_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    df = df[df[lag_cols].notna().all(axis=1)]
    X  = df[lag_cols]
    y  = gdp_actual.reindex(X.index)
    print(f"X_RF_bench ({n_lags} lags):    {X.shape[0]} quarters × {X.shape[1]} features")
    return X, y

def make_build_X(
    model_name: str,
    n_lags: int = 4,
    n_monthly_lags: int = 4,
    n_qd_lags: int = 4,
):
    """
    Returns a build_X function with the correct signature for the pipeline.
    
    Usage:
        build_fn = make_build_X("X1")
        X, y = build_fn(qd_filled, md_filled, gdp_cut, gdp_actual)
    """
    match model_name:
        case "X1":
            return lambda qd, md, gdp_cut, gdp_actual: build_X1_from_cut(qd, md, gdp_cut, gdp_actual, n_lags)
        case "X2":
            return lambda qd, md, gdp_cut, gdp_actual: build_X2_from_cut(qd, md, gdp_cut, gdp_actual, n_lags)
        case "X3":
            return lambda qd, md, gdp_cut, gdp_actual: build_X3_from_cut(qd, md, gdp_cut, gdp_actual, n_lags)
        case "X4":
            return lambda qd, md, gdp_cut, gdp_actual: build_X4_from_cut(qd, md, gdp_cut, gdp_actual, n_monthly_lags, n_qd_lags)
        case "X_AR":
            return lambda qd, md, gdp_cut, gdp_actual: build_X_AR_from_cut(gdp_cut, gdp_actual, n_lags)
        case "X_RF_bench":
            return lambda qd, md, gdp_cut, gdp_actual: build_X_RF_bench_from_cut(gdp_cut, gdp_actual, n_lags)
        case _:
            raise ValueError(f"Unknown model_name '{model_name}'. Must be one of: X1, X2, X3, X4, X_AR, X_RF_bench")