import numpy as np
from scipy import stats
import pandas as pd
import numpy as np
from itertools import combinations
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

def compare_model_matrix(df, time_col='quarter_date', **dm_kwargs):
    """
    Runs pairwise Diebold-Mariano tests across all (model, version) combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: model_name, model_version, y_actual, y_predicted, <time_col>
    time_col : str
        Column used to align observations across models (e.g. 'date', 'period').
    **dm_kwargs
        Passed directly to diebold_mariano() (e.g. loss='absolute', h=3).

    Returns
    -------
    matrix : pd.DataFrame
        N x N matrix of DM statistics. Negative = row model is better.
    pval_matrix : pd.DataFrame
        N x N matrix of p-values.
    skipped : list of str
        Pairs that were skipped and why.
    """
    df = df.copy()
    df['id'] = df['model_name'] + "_v" + df['version'].astype(str)

    # Pivot so each model-version is a column, rows are time periods
    # This enforces alignment on time_col automatically
    predictions = df.pivot_table(
        index=time_col,
        columns='id',
        values='nowcast',
        aggfunc='first'        # if duplicates exist, take first
    )

    # y_actual should be the same across all models — validate and extract once
    actuals = (
        df.drop_duplicates(subset=[time_col])
        .set_index(time_col)['gdp_actual']
        .reindex(predictions.index)
    )

    if actuals.isna().any():
        raise ValueError(f"Missing y_actual values for some {time_col} entries.")

    unique_ids = sorted(predictions.columns)
    results    = []
    skipped    = []

    for id1, id2 in combinations(unique_ids, 2):

        yhat1 = predictions[id1]
        yhat2 = predictions[id2]

        # Only use time periods where BOTH models have predictions
        mask = yhat1.notna() & yhat2.notna()
        n_overlap = mask.sum()

        if n_overlap < 10:
            skipped.append(f"{id1} vs {id2}: only {n_overlap} overlapping observations.")
            continue

        try:
            stat, pval = dm_test(
                actuals[mask].values,
                yhat1[mask].values,
                yhat2[mask].values,
                **dm_kwargs
            )
            results.append({'m1': id1, 'm2': id2, 'stat': stat,  'p': pval})
            results.append({'m1': id2, 'm2': id1, 'stat': -stat, 'p': pval})

        except Exception as e:
            skipped.append(f"{id1} vs {id2}: {e}")

    if not results:
        raise RuntimeError("No valid pairs found. Check your data alignment and grouping.")

    res_df = pd.DataFrame(results)

    matrix      = res_df.pivot(index='m1', columns='m2', values='stat')
    pval_matrix = res_df.pivot(index='m1', columns='m2', values='p')

    # Diagonal is undefined (model vs itself) — fill with NaN
    for m in unique_ids:
        if m in matrix.columns:
            matrix.loc[m, m]      = np.nan
            pval_matrix.loc[m, m] = np.nan

    # Align index and columns to the same order
    matrix      = matrix.loc[unique_ids, unique_ids]
    pval_matrix = pval_matrix.loc[unique_ids, unique_ids]

    if skipped:
        print(f"Skipped {len(skipped)} pairs:")
        for s in skipped: print(f"  - {s}")

    return matrix, pval_matrix, skipped

def dm_test(
    y_actual: np.ndarray,
    y_hat1: np.ndarray,
    y_hat2: np.ndarray,
    loss: str = "squared",
    h: int = 1,
    power: float = 2.0,
    bandwidth: str = "auto",
) -> tuple[float, float]:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    Tests H0: E[d_t] = 0, where d_t = L(e1_t) - L(e2_t) is the loss
    differential between forecast 1 and forecast 2.

    A negative test statistic favours forecast 1 (lower loss);
    a positive statistic favours forecast 2.

    Parameters
    ----------
    y_actual : array-like, shape (n,)
        Realised values.
    y_hat1 : array-like, shape (n,)
        Forecasts from model 1.
    y_hat2 : array-like, shape (n,)
        Forecasts from model 2.
    loss : {"squared", "absolute", "power"}
        Loss function applied to each forecast error.
        - "squared"  : L(e) = e²          (default, MSE-based)
        - "absolute" : L(e) = |e|          (MAE-based)
        - "power"    : L(e) = |e|^power    (generalised, set `power` as needed)
    h : int, default 1
        Forecast horizon. Used when bandwidth="fixed" to set HAC lags = h-1.
    power : float, default 2.0
        Exponent used when loss="power". Ignored for other loss choices.
    bandwidth : {"auto", "fixed"}
        HAC lag selection rule.
        - "auto"  : uses floor(P^(1/3)) — the data-driven rule (matches the slide).
        - "fixed" : uses h-1 — the theoretically motivated choice for an
                    h-step forecast error (MA(h-1) structure).

    Returns
    -------
    dm_stat : float
        The DM test statistic ~ N(0,1) asymptotically.
    p_value : float
        Two-sided p-value. Uses t(n-1) with Harvey-Leybourne-Newbold (1997)
        small-sample correction when n < 50, otherwise standard Normal.

    Raises
    ------
    ValueError
        If arrays differ in length, contain NaNs, or arguments are invalid.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> y   = rng.standard_normal(200)
    >>> h1  = y + rng.standard_normal(200) * 0.5   # better forecast
    >>> h2  = y + rng.standard_normal(200) * 1.0
    >>> dm_stat, p_val = diebold_mariano(y, h1, h2)
    >>> print(f"DM = {dm_stat:.4f}, p = {p_val:.4f}")
    """
    y_actual = np.asarray(y_actual, dtype=float)
    y_hat1   = np.asarray(y_hat1,   dtype=float)
    y_hat2   = np.asarray(y_hat2,   dtype=float)

    # ── Validation ────────────────────────────────────────────────────────────
    if not (y_actual.shape == y_hat1.shape == y_hat2.shape):
        raise ValueError("y_actual, y_hat1, and y_hat2 must have the same shape.")
    if np.any(np.isnan(y_actual) | np.isnan(y_hat1) | np.isnan(y_hat2)):
        raise ValueError("Inputs contain NaN values. Remove or impute before testing.")
    if h < 1:
        raise ValueError("Forecast horizon h must be >= 1.")
    if bandwidth not in {"auto", "fixed"}:
        raise ValueError(f"bandwidth must be 'auto' or 'fixed'; got '{bandwidth}'.")

    n = len(y_actual)

    # ── Loss differentials ────────────────────────────────────────────────────
    e1 = y_actual - y_hat1
    e2 = y_actual - y_hat2

    loss_fn = {
        "squared":  lambda e: e ** 2,
        "absolute": lambda e: np.abs(e),
        "power":    lambda e: np.abs(e) ** power,
    }
    if loss not in loss_fn:
        raise ValueError(f"loss must be one of {list(loss_fn)}; got '{loss}'.")

    d     = loss_fn[loss](e1) - loss_fn[loss](e2)
    d_bar = d.mean()

    # ── HAC lag length ────────────────────────────────────────────────────────
    if bandwidth == "auto":
        # P^(1/3) rule — data-driven, consistent with the slide
        n_lags = int(np.floor(n ** (1 / 3)))
    else:
        # h-1 rule — theoretically grounded for h-step-ahead forecasts
        n_lags = h - 1

    # ── HAC variance (Newey-West, Bartlett kernel) ────────────────────────────
    gamma_0 = np.var(d, ddof=0)
    hac_var  = gamma_0

    for k in range(1, n_lags + 1):
        gamma_k = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
        weight   = 1.0 - k / (n_lags + 1)    # Bartlett kernel
        hac_var += 2.0 * weight * gamma_k

    # Variance of the sample mean d̄
    var_d_bar = hac_var / n

    if var_d_bar <= 0:
        raise RuntimeError(
            "Estimated variance of the loss differential is non-positive. "
            "Check for degenerate or constant forecast errors."
        )

    dm_stat = d_bar / np.sqrt(var_d_bar)

    # ── Harvey-Leybourne-Newbold (1997) small-sample correction ──────────────
    # Rescales the statistic and uses t(n-1) when n < 50
    if n < 50:
        hln_factor = np.sqrt(
            (n + 1 - 2 * h + h * (h - 1) / n) / n
        )
        dm_stat *= hln_factor
        p_value  = 2.0 * stats.t.sf(np.abs(dm_stat), df=n - 1)
    else:
        p_value  = 2.0 * stats.norm.sf(np.abs(dm_stat))

    return float(dm_stat), float(p_value)



# Initialize connection
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
from database.client import get_backend_client


def fetch_forecast_data(table_name: str = "evaluation") -> pd.DataFrame:
    """
    Pulls all forecast data from Supabase and returns a cleaned DataFrame.
    """
    supabase = get_backend_client()
    response = supabase.table(table_name).select("quarter_date",
                                                 "version", 
                                                 "gdp_actual","AR_Benchmark","RF_Lags_Average",
                                                 "RF_Lags_UMIDAS", "LASSO_UMIDAS",
                                                 "LASSO_Average", "LASSO_Lags_Average",
                                                 "All_Model_Average").execute()
    
    # Convert to DataFrame
    df = pd.DataFrame(response.data)
    df_long = df.melt(
        id_vars=['quarter_date', 'version', 'gdp_actual'], 
        value_vars=["AR_Benchmark","RF_Lags_Average","RF_Lags_UMIDAS", "LASSO_UMIDAS",
                    "LASSO_Average", "LASSO_Lags_Average","All_Model_Average"],
        var_name='model_name', 
        value_name='nowcast'
)
    
    return df_long



# Usage
df_forecasts = fetch_forecast_data()
print(df_forecasts.head(15))

# ── TEST ──────────────────────────────────────────────────────────────────────
matrix, pval_matrix, skipped = compare_model_matrix(
    df_forecasts,
    time_col='quarter_date',
    loss='squared',
    bandwidth='auto'
)

MODEL_COLUMNS = [
    "AR_Benchmark",
    "RF_Lags_Average",
    "RF_Lags_UMIDAS",
    "LASSO_UMIDAS",
    "LASSO_Average",
    "LASSO_Lags_Average",
    "All_Model_Average",
]


# print("Test Statistic:" ,matrix)
# print("P-value:" ,pval_matrix)

from pipeline.load_data import save_df
save_df(matrix, "../data", "dm_stat_matrix.csv")
save_df(pval_matrix, "../data", "dm_pval_matrix.csv")

