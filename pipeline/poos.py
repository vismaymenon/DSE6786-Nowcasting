import numpy as np
import pandas as pd
from typing import Callable
from datetime import date
import os
from dotenv import load_dotenv
from output_x import load_filled_data, build_X1, build_X2, build_X3, build_X4, build_X_AR, build_X_RF_bench
from ragged_edge import fill_ragged_edge_until

# ── Global constants ────────────────────────────────────────────────────────────
TRAIN_SIZE = 162   # Fixed training window size (quarters). Kept constant so the
                   # DM-test evaluation window is always the same regardless of
                   # how many future quarters of data are added later.

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
    Cuts QD and MD to what would be available at prediction time, then fills
    the ragged edge with AR(p) forecasts up to q_predicted.

    Parameters
    ----------
    version     : int            — 1 to 6, representing which month within the
                                   quarter surrounding q_predicted we stand at
    q_predicted : pd.Timestamp   — quarter being predicted (e.g. 2025-03-01 for Q1 2025)
    QD_t        : pd.DataFrame   — full quarterly data (sasdate column)
    MD_t        : pd.DataFrame   — full monthly data (sasdate column)

    Returns
    -------
    qd_filled   : pd.DataFrame   — quarterly data filled up to q_predicted
    md_filled   : pd.DataFrame   — monthly data filled up to q_predicted
    gdp_cutoff  : pd.Timestamp   — last GDP quarter available at prediction time
    """

    # q_predicted label is the last month of the predicted quarter (e.g. 2025-03-01)
    q_start  = q_predicted - relativedelta(months=2)   # e.g. 2025-01-01
    prev_q_end = q_start - relativedelta(months=1)      # e.g. 2024-12-01

    version_cutoffs = {
        # version: (qd_cutoff,                             md_cutoff,                              gdp_cutoff)
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

    QD_cut = QD_t[QD_t["sasdate"] <= pd.Timestamp(qd_cutoff)].copy()
    MD_cut = MD_t[MD_t["sasdate"] <= pd.Timestamp(md_cutoff)].copy()

    qd_filled, md_filled = fill_ragged_edge_until(QD_cut, MD_cut, cutoff_date=q_predicted)

    return qd_filled, md_filled, gdp_cutoff

# ── POOS ──────────────────────────────────────────────────────────────────────


def poos_validation(
    method: Callable,
    buildX: Callable,
    QD_t: pd.DataFrame,
    MD_t: pd.DataFrame,
    y_full: pd.Series,
    version: int,
    num_test: int = 100,
) -> tuple[pd.DataFrame, float, float]:
    """
    Realistic pseudo out-of-sample validation.

    For each of the last num_test quarters with known GDP, we simulate standing
    at the prediction point in time (determined by version) by:
      1. Cutting QD/MD/GDP to only what would be published at that point.
      2. Filling the ragged edge with AR(p) forecasts up to q_predicted.
      3. Rebuilding the feature matrix from the filled snapshot.
      4. Training on all rows before q_predicted, predicting q_predicted.

    The test set excludes the current nowcast quarter (y=NaN) and any future
    quarters, so test_quarters = last num_test entries of y_full where y is known.

    Parameters
    ----------
    method      : Callable  — model function with signature method(X, y) → dict
                              where X/y last row is the test observation
    buildX      : Callable  — feature builder with signature buildX(df_md, df_qd) → (X, y)
                              must be one of build_X1 / build_X2 / build_X3 / build_X4
    QD_t        : pd.DataFrame — full quarterly data (sasdate column, unfiltered)
    MD_t        : pd.DataFrame — full monthly data (sasdate column, unfiltered)
    y_full      : pd.Series    — full GDP series indexed by quarter date;
                                 NaN for unreleased quarters
    version     : int          — 1–6, which month within the quarter we stand at
    num_test    : int          — number of OOS test quarters (default 100)

    Returns
    -------
    y_df  : pd.DataFrame — predictions and CIs for each test quarter
    rmse  : float
    mae   : float
    """
    # ── Identify test quarters ────────────────────────────────────────────────
    # Only evaluate on quarters where GDP is officially released (y is known).
    # This naturally excludes the current nowcast quarter (y=NaN).
    y_known = y_full[y_full.notna()]
    if len(y_known) < num_test:
        raise ValueError(f"Only {len(y_known)} known GDP quarters, need {num_test}.")
    test_quarters = y_known.index[-num_test:]

    records = []

    for i, q_predicted in enumerate(test_quarters):
        print(f"\n[{i+1}/{num_test}] q_predicted = {q_predicted.date()}")

        # ── Step 1: Cut and fill ──────────────────────────────────────────────
        qd_filled, md_filled, gdp_cutoff = cut_and_fill(
            version, q_predicted, QD_t, MD_t
        )

        # ── Step 2: Build feature matrix from filled snapshot ─────────────────
        # buildX loads GDP from DB internally; we truncate the returned X/y to
        # rows <= q_predicted to avoid any look-ahead beyond the test point.
        X, y = buildX(md_filled, qd_filled)
        X = X[X.index <= q_predicted]
        y = y[y.index <= q_predicted]

        if q_predicted not in X.index:
            print(f"  WARNING: {q_predicted.date()} missing from feature matrix — skipping.")
            continue

        # ── Step 3: Mask GDP lag columns beyond gdp_cutoff ───────────────────
        # GDP lag columns in X are named gdp_lag1 … gdp_lagN.
        # lag k for q_predicted refers to GDP at q_predicted - k quarters.
        # If that date > gdp_cutoff, the lag wasn't published yet → set to NaN.
        gdp_lag_cols = [c for c in X.columns if c.startswith("gdp_lag")]
        for col in gdp_lag_cols:
            k = int(col.replace("gdp_lag", ""))
            lag_date = q_predicted - pd.DateOffset(months=3 * k)
            if lag_date > pd.Timestamp(gdp_cutoff):
                X.loc[q_predicted, col] = np.nan

        if X.loc[q_predicted].isna().any():
            print(f"  WARNING: NaN in test row features for {q_predicted.date()} — skipping.")
            continue

        # ── Step 4: Split train / test ────────────────────────────────────────
        # Use the TRAIN_SIZE rows immediately before q_predicted (fixed window).
        X_before = X[X.index < q_predicted]
        y_before = y[y.index < q_predicted]

        valid_train = y_before.notna()
        X_before = X_before[valid_train]
        y_before = y_before[valid_train]

        X_train = X_before.iloc[-TRAIN_SIZE:]
        y_train = y_before.iloc[-TRAIN_SIZE:]

        if len(X_train) == 0:
            print(f"  WARNING: empty training set for {q_predicted.date()} — skipping.")
            continue

        X_window = pd.concat([X_train, X.loc[[q_predicted]]])
        y_window = pd.concat([y_train, y.loc[[q_predicted]]])

        # ── Step 5: Fit and predict ───────────────────────────────────────────
        result = method(X_window, y_window)
        _, y_train_actual, y_train_predicted, _, y_test_actual, y_test_predicted = result.values()
        train_rmse = float(np.sqrt(np.mean(
            (np.array(y_train_actual) - np.array(y_train_predicted)) ** 2
        )))

        records.append({
            "index":          q_predicted,
            "y_true":         float(y_test_actual),
            "y_hat":          float(y_test_predicted),
            "pred_50_lower":  float(y_test_predicted) - 0.674 * train_rmse,
            "pred_50_upper":  float(y_test_predicted) + 0.674 * train_rmse,
            "pred_80_lower":  float(y_test_predicted) - 1.282 * train_rmse,
            "pred_80_upper":  float(y_test_predicted) + 1.282 * train_rmse,
        })

    if not records:
        raise RuntimeError("No test quarters were successfully evaluated.")

    y_df = pd.DataFrame(records).set_index("index")
    rmse = float(np.sqrt(np.mean((y_df["y_true"] - y_df["y_hat"]) ** 2)))
    mae  = float(np.mean(np.abs(y_df["y_true"] - y_df["y_hat"])))
    print(f"\nPOOS complete — {len(y_df)} quarters | RMSE={rmse:.4f} | MAE={mae:.4f}")

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
    qd = pd.read_csv("data/filled_qd.csv")
    md = pd.read_csv("data/filled_md.csv")

    qd["sasdate"] = pd.to_datetime(qd["sasdate"], errors="coerce")
    md["sasdate"] = pd.to_datetime(md["sasdate"], errors="coerce")

    # Smoke-test cut_and_fill
    filled_qd, filled_md, gdp_cutoff = cut_and_fill(
        version=3,
        q_predicted=pd.Timestamp("2025-12-01"),
        QD_t=qd,
        MD_t=md,
    )
    print("QD tail:"); print(filled_qd.tail())
    print("MD tail:"); print(filled_md.tail())
    print(f"GDP cutoff: {gdp_cutoff.date()}")

    
