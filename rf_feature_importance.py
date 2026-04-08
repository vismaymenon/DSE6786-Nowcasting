"""
rf_feature_importance.py
========================
Generates Random Forest feature importance bar plots for the latest quarter
prediction, replicating the nowcast_single_latest() training window from
pipeline/prediction.py.

Usage:
    python rf_feature_importance.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from database.client import get_backend_client
from pipeline.output_x import load_filled_data, build_X2, build_X4

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
    Main entry point. Loads data, trains RF models for the latest quarter,
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

    print("\nDone.")


if __name__ == "__main__":
    generate_rf_importance_plots()
