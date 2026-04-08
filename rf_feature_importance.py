"""
rf_feature_importance.py
========================
Generates Random Forest feature importance bar plots and LASSO coefficient
bar plots for the latest quarter prediction, replicating the
nowcast_single_latest() training window from pipeline/prediction.py.

RF models:    RF_Lags_Average (X2), RF_Lags_UMIDAS (X4)
LASSO models: LASSO_Average (X1), LASSO_Lags_Average (X2), LASSO_UMIDAS (X3)

Usage:
    python rf_feature_importance.py
"""

import os
import pandas as pd
import numpy as np
import hdmpy as hd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from database.client import get_backend_client
from pipeline.output_x import load_filled_data, build_X1, build_X2, build_X3, build_X4

# ── RF hyperparameters (must match pipeline/models/rf.py) ────────────────────
N_ESTIMATORS = 1000
MAX_FEATURES = 0.3
RANDOM_STATE = 42

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "pipeline", "plots")
TRAIN_SIZE = 162
TOP_N = 20


def _load_gdp() -> pd.DataFrame:
    """Fetch GDP table from Supabase, ordered by date ascending."""
    client = get_backend_client()
    response = (
        client.table("gdp")
        .select("sasdate, GDPC1_t")
        .order("sasdate", desc=False)
        .execute()
    )
    gdp_df = pd.DataFrame(response.data)
    gdp_df["sasdate"] = pd.to_datetime(gdp_df["sasdate"])
    gdp_df = gdp_df.set_index("sasdate")
    return gdp_df


def _fill_prev_quarter(y: pd.Series, prev_idx, model_name: str) -> pd.Series:
    """
    If y at prev_idx is NaN, fill it with the latest Supabase nowcast for
    that quarter (same logic as nowcast_single_latest). Falls back to the
    series mean if no forecast is found.
    """
    y_filled = y.copy()
    if not pd.isna(y.loc[prev_idx]):
        return y_filled

    quarter_date = (
        pd.Period(prev_idx, freq="Q")
        .to_timestamp(how="end")
        .to_period("M")
        .to_timestamp()
        .date()
        .isoformat()
    )

    client = get_backend_client()
    response = (
        client.table("model_forecasts")
        .select("nowcast, month_date, run_date")
        .eq("model_name", model_name)
        .eq("quarter_date", quarter_date)
        .order("run_date", desc=True)
        .order("month_date", desc=True)
        .limit(1)
        .execute()
    )

    if response.data:
        fetched = float(response.data[0]["nowcast"])
        print(
            f"  Filling gdp.index[-2] ({prev_idx}) with Supabase nowcast: "
            f"{fetched:.4f} (month: {response.data[0]['month_date']}, "
            f"run: {response.data[0]['run_date']})"
        )
        y_filled.loc[prev_idx] = fetched
    else:
        fallback = float(y.dropna().mean())
        print(
            f"  No forecast found for {prev_idx} in Supabase — "
            f"falling back to mean: {fallback:.4f}"
        )
        y_filled.loc[prev_idx] = fallback

    return y_filled


def train_rf_for_latest_quarter(X: pd.DataFrame, y: pd.Series, gdp: pd.DataFrame, model_name: str):
    """
    Replicates the nowcast_single_latest() training window for the latest
    quarter (gdp.index[-1]), trains a RandomForestRegressor, and returns the
    fitted model along with the training feature names.

    Parameters
    ----------
    X : pd.DataFrame  — full feature matrix (all quarters)
    y : pd.Series     — GDP growth series aligned to X
    gdp : pd.DataFrame — GDP DataFrame (index = quarter dates)
    model_name : str  — used for the NaN-fill Supabase lookup

    Returns
    -------
    rf : fitted RandomForestRegressor
    feature_names : list[str]
    target_idx : the latest quarter label
    """
    target_idx = gdp.index[-1]
    prev_idx = gdp.index[-2]

    # Fill previous quarter if NaN (same as nowcast_single_latest)
    y_filled = _fill_prev_quarter(y, prev_idx, model_name)

    # Construct the same rolling window used by nowcast_single_latest
    target_pos = X.index.get_loc(target_idx)
    window_start = target_pos - TRAIN_SIZE
    X_window = X.iloc[window_start : target_pos + 1]
    y_window = y_filled.iloc[window_start : target_pos + 1]

    # Split into train / test (last row = test, as in randomForest())
    X_train = X_window.iloc[:-1]
    y_train = y_window.iloc[:-1].values

    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_features=MAX_FEATURES,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train.values, y_train)

    return rf, list(X_window.columns), target_idx


def plot_feature_importance(rf, feature_names: list, target_idx, model_name: str):
    """
    Generate and save a horizontal bar plot of the top-N most important
    features for the given fitted RF model.

    Also prints the top features to stdout.

    Parameters
    ----------
    rf : fitted RandomForestRegressor
    feature_names : list[str]
    target_idx : quarter label (used in plot title)
    model_name : str
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    top_indices = indices[:TOP_N]
    top_names = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]

    # Print to stdout
    quarter_label = pd.Period(target_idx, freq="Q")
    print(f"\nTop {TOP_N} features — {model_name} (Latest Quarter: {quarter_label}):")
    for rank, (name, imp) in enumerate(zip(top_names, top_importances), start=1):
        print(f"  {rank:>2}. {name:<50s}  {imp:.6f}")

    # Reverse for horizontal bar plot (most important at top)
    plot_names = top_names[::-1]
    plot_importances = top_importances[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(TOP_N), plot_importances, align="center", color="steelblue")
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels(plot_names, fontsize=9)
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)")
    ax.set_title(f"RF Feature Importance — {model_name}\n(Latest Quarter: {quarter_label})")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
    plt.tight_layout()

    filename = f"rf_importance_{model_name}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to: {filepath}")


def generate_rf_importance_plots():
    """
    Loads data, trains RF models for the latest quarter,
    and saves feature importance plots.
    """
    print("Loading filled data …")
    df_md, df_qd = load_filled_data()

    print("\nBuilding feature matrices …")
    X2, y2 = build_X2(df_md, df_qd, n_lags=4)
    X4, y4 = build_X4(df_md, df_qd, n_monthly_lags=4, n_qd_lags=4)

    print("\nFetching GDP from Supabase …")
    gdp = _load_gdp()

    models = [
        ("RF_Lags_Average", X2, y2),
        ("RF_Lags_UMIDAS",  X4, y4),
    ]

    for model_name, X, y in models:
        print(f"\n{'='*60}")
        print(f"Training {model_name} for latest quarter …")
        rf, feature_names, target_idx = train_rf_for_latest_quarter(
            X, y, gdp, model_name
        )
        plot_feature_importance(rf, feature_names, target_idx, model_name)


def train_lasso_for_latest_quarter(X: pd.DataFrame, y: pd.Series, gdp: pd.DataFrame, model_name: str):
    """
    Replicates the nowcast_single_latest() training window for the latest
    quarter (gdp.index[-1]), runs hdmpy.rlasso, and returns the
    non-zero coefficients with their feature names.

    Parameters
    ----------
    X : pd.DataFrame   — full feature matrix (all quarters)
    y : pd.Series      — GDP growth series aligned to X
    gdp : pd.DataFrame — GDP DataFrame (index = quarter dates)
    model_name : str   — used for the NaN-fill Supabase lookup

    Returns
    -------
    coef_series : pd.Series  — non-zero coefficients, sorted by absolute value
    target_idx  : the latest quarter label
    """
    target_idx = gdp.index[-1]
    prev_idx = gdp.index[-2]

    # Fill previous quarter if NaN (same as nowcast_single_latest)
    y_filled = _fill_prev_quarter(y, prev_idx, model_name)

    # Construct the same rolling window used by nowcast_single_latest
    target_pos = X.index.get_loc(target_idx)
    window_start = target_pos - TRAIN_SIZE
    X_window = X.iloc[window_start : target_pos + 1]
    y_window = y_filled.iloc[window_start : target_pos + 1]

    # Train on all but last row (last = test point, same as fit_lasso)
    X_train = X_window.iloc[:-1].copy()
    y_train = y_window.iloc[:-1].copy()

    # Mirror fit_lasso() cleaning steps
    col_std = X_train.std()
    zero_var_cols = col_std[col_std == 0].index.tolist()
    if zero_var_cols:
        print(f"  Dropping zero-variance columns: {zero_var_cols}")
        X_train = X_train.drop(columns=zero_var_cols)

    nan_counts = X_train.isna().sum()
    mostly_nan_cols = nan_counts[nan_counts > 0.9 * len(X_train)].index.tolist()
    if mostly_nan_cols:
        print(f"  Dropping mostly-NaN columns: {mostly_nan_cols}")
        X_train = X_train.drop(columns=mostly_nan_cols)

    mask_valid_rows = X_train.notna().all(axis=1)
    if not mask_valid_rows.all():
        print(f"  Dropping {(~mask_valid_rows).sum()} training rows with NaNs")
        X_train = X_train[mask_valid_rows]
        y_train = y_train[mask_valid_rows]

    if X_train.empty:
        raise ValueError(f"No non-NaN rows/columns left in X_train for {model_name}.")

    model = hd.rlasso(X_train, y_train.values, post=True, homoskedastic=False)
    coefs = np.nan_to_num(np.array(model.est["coefficients"]).flatten())
    coefs = coefs[1:]  # drop intercept

    coef_series = pd.Series(coefs, index=X_train.columns)
    coef_series = coef_series[coef_series != 0].sort_values(key=np.abs, ascending=False)

    return coef_series, target_idx


def plot_lasso_coefficients(coef_series: pd.Series, target_idx, model_name: str):
    """
    Print and save a horizontal bar plot of the LASSO non-zero coefficients,
    sorted by absolute value (largest at top).

    Parameters
    ----------
    coef_series : pd.Series — non-zero LASSO coefficients
    target_idx  : quarter label (used in plot title)
    model_name  : str
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    quarter_label = pd.Period(target_idx, freq="Q")

    # Print to stdout
    print(f"\nNon-zero LASSO coefficients — {model_name} (Latest Quarter: {quarter_label}):")
    if coef_series.empty:
        print("  (all coefficients are zero)")
    else:
        for rank, (name, coef) in enumerate(coef_series.items(), start=1):
            print(f"  {rank:>2}. {name:<50s}  {coef:+.6f}")

    if coef_series.empty:
        print("  Skipping plot — no non-zero coefficients.")
        return

    # Reverse for horizontal bar (largest at top)
    plot_series = coef_series.iloc[::-1]
    colors = ["steelblue" if v > 0 else "tomato" for v in plot_series]

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_series) * 0.35)))
    bars = ax.barh(range(len(plot_series)), plot_series.values, align="center", color=colors)
    ax.set_yticks(range(len(plot_series)))
    ax.set_yticklabels(plot_series.index, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("LASSO Coefficient")
    ax.set_title(f"LASSO Coefficients — {model_name}\n(Latest Quarter: {quarter_label})")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
    plt.tight_layout()

    filename = f"lasso_coefficients_{model_name}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to: {filepath}")


def generate_lasso_coefficient_plots():
    """
    Loads data, fits LASSO models for the latest quarter,
    and saves coefficient bar plots.
    """
    print("Loading filled data …")
    df_md, df_qd = load_filled_data()

    print("\nBuilding feature matrices …")
    X1, y1 = build_X1(df_md, df_qd)
    X2, y2 = build_X2(df_md, df_qd, n_lags=4)
    X3, y3 = build_X3(df_md, df_qd)

    print("\nFetching GDP from Supabase …")
    gdp = _load_gdp()

    models = [
        ("LASSO_Average",      X1, y1),
        ("LASSO_Lags_Average", X2, y2),
        ("LASSO_UMIDAS",       X3, y3),
    ]

    for model_name, X, y in models:
        print(f"\n{'='*60}")
        print(f"Fitting {model_name} for latest quarter …")
        coef_series, target_idx = train_lasso_for_latest_quarter(
            X, y, gdp, model_name
        )
        plot_lasso_coefficients(coef_series, target_idx, model_name)


if __name__ == "__main__":
    print("=" * 60)
    print("RF Feature Importance")
    print("=" * 60)
    generate_rf_importance_plots()

    print("\n" + "=" * 60)
    print("LASSO Coefficients")
    print("=" * 60)
    generate_lasso_coefficient_plots()

    print("\nDone.")
