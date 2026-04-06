import numpy as np
import pandas as pd
from typing import Callable
from datetime import date
import os
from dotenv import load_dotenv
from output_x import load_filled_data, build_X1, build_X2, build_X3, build_X4, build_X_AR, build_X_RF_bench
from ragged_edge import fill_ragged_edge_until

# ── Placeholder model ───────────────────────────────────────────────────────────

def placeholder_model(X, y):
    """
    Benchmark (unconditional mean) model.
    Treats the last row of X and last element of y as the test observation.

    Inputs:
        X : pd.DataFrame, shape (t+1, n_features)  — last row is test
        y : pd.Series,    shape (t+1,)              — last element is test

    Outputs:
        X_train          (pd.DataFrame) : training X
        y_actual         (np.ndarray)   : training y
        y_train_predicted(np.ndarray)   : in-sample predictions (all = mean)
        X_test           (pd.DataFrame) : test X (last row of input X)
        y_test_actual    (float)        : held-out true value
        y_test_predicted (float)        : prediction = mean of training y
    """
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1].values
    X_test = X.iloc[[-1]]
    y_test_actual = float(y.iloc[-1])

    y_train_mean = float(np.mean(y_train))
    y_train_predicted = y_train_mean
    y_test_predicted  = y_train_mean

    return {
        "X_train": X_train,
        "y_train": y_train,
        "y_train_predicted": y_train_predicted,
        "X_test": X_test,
        "y_test_actual": y_test_actual,
        "y_test_predicted": y_test_predicted
    }


# -- Inside function for POOS --------------------------------------------------

from dateutil.relativedelta import relativedelta

def cut_and_fill(version: int,
                 q_predicted: pd.Timestamp,
                 QD_t: pd.DataFrame,
                 MD_t: pd.DataFrame
                 ):
    """
    Cuts QD, MD, and y to what would be available at prediction time.

    Parameters
    ----------
    version     : int          — 1 to 6, representing month of prediction
                                 within the two quarters surrounding q_predicted
    q_predicted : pd.Timestamp — quarter being predicted (e.g. 2025-03-01 for Q1 2025)
    build       : Callable     — feature builder e.g. build_X1, build_X3
    QD_t        : pd.DataFrame — full quarterly data (sasdate index)
    MD_t        : pd.DataFrame — full monthly data (sasdate index)
    y_t         : pd.Series    — full GDP series
    train_size  : int          — number of quarters to use for training

    Returns
    -------
    X : pd.DataFrame — feature matrix up to available data
    y : pd.Series    — GDP up to available data
    """

    # ── Determine cutoff dates based on version ───────────────────────────────
    # q_predicted = first month of the last month of the predicted quarter
    # e.g. Q1 2025 → 2025-03-01

    # Quarter start = 2 months before q_predicted label
    q_start = q_predicted - relativedelta(months=2)  # e.g. 2025-01-01

    # Previous quarter end
    prev_q_end = q_start - relativedelta(months=1)   # e.g. 2024-12-01
    prev_q_end_qd = prev_q_end                        # QD uses quarter-end dates

    version_cutoffs = {
        # version: (qd_cutoff,                    md_cutoff,                         gdp_cutoff)
        1: (prev_q_end - relativedelta(months=3),  prev_q_end - relativedelta(months=2), prev_q_end - relativedelta(months=3)),
        2: (prev_q_end - relativedelta(months=3),  prev_q_end - relativedelta(months=1), prev_q_end - relativedelta(months=3)),
        3: (prev_q_end,                            prev_q_end,                            prev_q_end),
        4: (prev_q_end,                            q_start,                               prev_q_end),
        5: (prev_q_end,                            q_start + relativedelta(months=1),     prev_q_end),
        6: (q_predicted,                           q_predicted,                           prev_q_end),
    }

    qd_cutoff, md_cutoff, gdp_cutoff = version_cutoffs[version]

    print(f"Version {version} | Predicting: {q_predicted.date()} | "
          f"QD until: {qd_cutoff.date()} | "
          f"MD until: {md_cutoff.date()} | "
          f"GDP until: {gdp_cutoff.date()}")

    # ── Cut data to cutoff dates ───────────────────────────────────────────────
    QD_cut = QD_t[QD_t["sasdate"] <= pd.Timestamp(qd_cutoff)].copy()
    MD_cut = MD_t[MD_t["sasdate"] <= pd.Timestamp(md_cutoff)].copy()

    # ── Build feature matrix using the build function ─────────────────────────
    qd_filled, md_filled = fill_ragged_edge_until(QD_cut, MD_cut, cutoff_date=q_predicted)

    return qd_filled, md_filled

# ── POOS ──────────────────────────────────────────────────────────────────────


def poos_validation(
    method: Callable,
    buildX: Callable,
    QD_t: pd.DataFrame, 
    MD_t: pd.DataFrame,
    y_t: pd.Series,
    version: int,
    num_train: int = 163,
    num_test: int = 100,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:

    n = len(y)
    train_size = n - num_test
    test_indices, actuals = [], []
    preds_point, preds_50_lower, preds_50_upper, preds_80_lower, preds_80_upper = [], [], [], [], []

    for t in range(num_test):
        q_predicted = y.index[train_size + t]  # quarter being predicted
        QD_filled, MD_filled = cut_and_fill(version, q_predicted, buildX, QD_t, MD_t, y_t, num_train)
        
        
        X_window = X_known.iloc[t:t+train_size+1]
        y_window = y_known.iloc[t:t+train_size+1]

        _, y_train_actual, y_train_predicted, _, y_test_actual, y_test_predicted = method(X_window, y_window).values()
        std_error = np.std(y_train_actual - y_train_predicted)

        test_indices.append(y_known.index[t + train_size])
        actuals.append(float(y_test_actual))
        preds_point.append(float(y_test_predicted))
        preds_50_lower.append(float(y_test_predicted) - 0.674 * std_error)
        preds_50_upper.append(float(y_test_predicted) + 0.674 * std_error)
        preds_80_lower.append(float(y_test_predicted) - 1.282 * std_error)
        preds_80_upper.append(float(y_test_predicted) + 1.282 * std_error)

    y_df = pd.DataFrame(
        index=test_indices,
        data={
            "y_true": actuals,
            "y_hat": preds_point,
            "pred_50_lower": preds_50_lower,
            "pred_50_upper": preds_50_upper,
            "pred_80_lower": preds_80_lower,
            "pred_80_upper": preds_80_upper,
        }
    )

    rmse = np.sqrt(np.mean((y_df["y_true"] - y_df["y_hat"]) ** 2))
    mae  = np.mean(np.abs(y_df["y_true"] - y_df["y_hat"]))

    return y_df, rmse, mae


# Plot results 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_poos_results(y_full, y_df, title="POOS Forecast vs Actual", last_n=200):
    fig, ax = plt.subplots(figsize=(14, 5))

    # Trim full series to last n points 
    y_plot = y_full.iloc[-last_n:]
    cutoff_date = y_plot.index[0]

    ax.plot(y_plot.index, y_plot.values, color="black", linewidth=1.2,
            label="Actual (full sample)", zorder=3)

    # ── Filter y_df to only dates within the last_n window ───────────────────
    y_df_plot = y_df[y_df.index >= cutoff_date]
    idx = y_df_plot.index

    ax.plot(idx, y_df_plot["y_hat"], color="red", linewidth=1.2,
            label="Predicted (OOS)", zorder=4)

    ax.fill_between(idx, y_df_plot["pred_50_lower"], y_df_plot["pred_50_upper"],
                    alpha=0.4, color="steelblue", label="50% CI")

    ax.fill_between(idx, y_df_plot["pred_80_lower"], y_df_plot["pred_80_upper"],
                    alpha=0.2, color="steelblue", label="80% CI")

    ax.axvline(x=idx[0], color="grey", linestyle=":", linewidth=1, label="OOS start")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("GDP Growth Rate (%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()   # ← rotates date labels so they don't overlap
    plt.tight_layout()

    os.makedirs("pipeline/plots", exist_ok=True)
    safe_title = title.replace(" ", "_").replace("/", "_")
    fig.savefig(os.path.join("pipeline/plots", f"{safe_title}.png"), dpi=300, bbox_inches="tight")
    
    plt.close()

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    qd = pd.read_csv("data/filled_qd.csv")[:-5]
    md = pd.read_csv("data/filled_md.csv")[:-5]

    # Ensure datetime for sasdate
    qd["sasdate"] = pd.to_datetime(qd["sasdate"], errors="coerce")
    md["sasdate"] = pd.to_datetime(md["sasdate"], errors="coerce")

    filled_qd, filled_md = cut_and_fill(
        1,
        pd.Timestamp("2025-12-01"),
        qd,
        md,
        pd.Series(dtype="float64", index=pd.to_datetime([])),  # dummy y_t for now
    )
    print(filled_qd.tail())
    print(filled_md.tail())
    
    # Use lags of y as a simple feature matrix (AR-style)
    X_df = pd.DataFrame({
        "lag_1": y_series.shift(1),
        "lag_2": y_series.shift(2),
        "lag_3": y_series.shift(3),
    })

    # Align and drop NaNs
    df = pd.concat([X_df, y_series], axis=1).dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    print(f"Sample size: {len(y)}")
    print(f"Features:    {X.columns.tolist()}\n")

    # Run POOS with placeholder model
    X_out, y_out, rmse, mae = poos_validation(
        method=placeholder_model,
        X=X,
        y=y,
        num_test=100,
    )

    plot_poos_results(y, y_out, title="INDPRO — Benchmark Model POOS")

    print("=== POOS Results (first 5 rows) ===")
    print(y_out.head())

    rmse = np.sqrt(np.mean((y_out["y_true"] - y_out["y_hat"]) ** 2))
    mae  = np.mean(np.abs(y_out["y_true"] - y_out["y_hat"]))
    print(f"\nOut-of-sample RMSE : {rmse:.6f}")
    print(f"Out-of-sample MAE  : {mae:.6f}")
    print(f"\nOOS observations   : {len(y_out)}")

    
