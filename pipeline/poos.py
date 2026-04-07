import numpy as np
import pandas as pd
from typing import Callable
from datetime import date
import os
from dotenv import load_dotenv
from output_x_poos import make_build_X
from ragged_edge import fill_ragged_edge_until
from database.client import get_backend_client

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
                 MD_t: pd.DataFrame,
                 gdp: pd.Series,
                 model_name: str = "All_Model_Average",
                 ):

    client = get_backend_client()   

    q_start    = q_predicted - relativedelta(months=2)
    prev_q_end = q_start - relativedelta(months=1)

    version_cutoffs = {
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

    # ── Cut GDP at gdp_cutoff, then fill gap up to prev_q_end with nowcasts ──
    gdp_cut = gdp[gdp.index <= pd.Timestamp(gdp_cutoff)].copy()

    quarters_to_fill = pd.date_range(
        start=gdp_cutoff + relativedelta(months=3),
        end=prev_q_end,
        freq="QS-DEC",
    )

    if len(quarters_to_fill) > 0:
        quarter_date_strs = [
            pd.Period(q, freq="Q").to_timestamp(how="end").to_period("M").to_timestamp().date().isoformat()
            for q in quarters_to_fill
        ]

        response = (
            client.table("model_forecasts")
            .select("quarter_date, nowcast, run_date")
            .eq("model_name", model_name)
            .in_("quarter_date", quarter_date_strs)
            .order("run_date", desc=True)
            .execute()
        )

        if not response.data:
            print(f"  Warning: no forecasts found to fill GDP gap — leaving as NaN.")
        else:
            fetched = (
                pd.DataFrame(response.data)
                .sort_values("run_date", ascending=False)
                .drop_duplicates(subset="quarter_date")
                .set_index("quarter_date")
            )

            for q, q_date_str in zip(quarters_to_fill, quarter_date_strs):
                if q_date_str in fetched.index:
                    nowcast = float(fetched.loc[q_date_str, "nowcast"])
                    gdp_cut.loc[q] = nowcast
                    print(f"  Filled GDP at {q.date()} with nowcast: {nowcast:.4f}")
                else:
                    print(f"  Warning: no forecast found for {q.date()} — leaving as NaN.")

    return qd_filled, md_filled, gdp_cut

# ── POOS ──────────────────────────────────────────────────────────────────────

def poos_validation(
    method: Callable,
    buildname: str,
    QD_t: pd.DataFrame,
    MD_t: pd.DataFrame,
    y_full: pd.Series,
    version: int,
    model_name: str = "All_Model_Average",
    num_test: int = 100,
) -> tuple[pd.DataFrame, float, float]:
    """
    Realistic pseudo out-of-sample validation.

    For each of the last num_test quarters with known GDP, we simulate standing
    at the prediction point in time (determined by version) by:
      1. Cutting QD/MD/GDP to only what would be published at that point.
      2. Filling the ragged edge with AR(p) forecasts up to q_predicted.
      3. Filling GDP gap with nowcasts from Supabase.
      4. Rebuilding the feature matrix from the filled snapshot.
      5. Training on all rows before q_predicted, predicting q_predicted.

    Parameters
    ----------
    method      : Callable     — model function with signature method(X, y) → dict
    buildname   : str          — name of feature builder, e.g. "X1", "X2", "X_AR"
                             passed to make_build_X() internally
    QD_t        : pd.DataFrame — full quarterly data (sasdate column, unfiltered)
    MD_t        : pd.DataFrame — full monthly data (sasdate column, unfiltered)
    y_full      : pd.Series    — full GDP series indexed by quarter date
    version     : int          — 1–6, which month within the quarter we stand at
    client                     — Supabase client
    model_name  : str          — model to use for GDP gap filling (default: All_Model_Average)
    num_test    : int          — number of OOS test quarters (default 100)

    Returns
    -------
    y_df  : pd.DataFrame — predictions and CIs for each test quarter
    rmse  : float
    mae   : float
    """
    # ── Identify test quarters ────────────────────────────────────────────────
    y_known = y_full[y_full.notna()]
    if len(y_known) < num_test:
        raise ValueError(f"Only {len(y_known)} known GDP quarters, need {num_test}.")
    test_quarters = y_known.index[-num_test:]

    records = []

    buildX = make_build_X(buildname)

    for i, q_predicted in enumerate(test_quarters):
        print(f"\n[{i+1}/{num_test}] q_predicted = {q_predicted.date()}")

        # ── Step 1: Cut and fill ──────────────────────────────────────────────
        qd_filled, md_filled, gdp_cut = cut_and_fill(
            version=version,
            q_predicted=q_predicted,
            QD_t=QD_t,
            MD_t=MD_t,
            gdp=y_full,
            model_name=model_name,
        )

        # ── Step 2: Build feature matrix from filled snapshot ─────────────────
        X, y = buildX(qd_filled, md_filled, gdp_cut, y_full)
        X = X[X.index <= q_predicted]
        y = y[y.index <= q_predicted]

        if q_predicted not in X.index:
            print(f"  WARNING: {q_predicted.date()} missing from feature matrix — skipping.")
            continue

        if X.loc[q_predicted].isna().any():
            print(f"  WARNING: NaN in test row features for {q_predicted.date()} — skipping.")
            continue

        # ── Step 3: Split train / test ────────────────────────────────────────
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

        # ── Step 4: Fit and predict ───────────────────────────────────────────
        result = method(X_window, y_window)
        _, y_train_actual, y_train_predicted, _, y_test_actual, y_test_predicted = result.values()
        train_rmse = float(np.sqrt(np.mean(
            (np.array(y_train_actual) - np.array(y_train_predicted)) ** 2
        )))

        records.append({
            "index":         q_predicted,
            "y_true":        float(y_test_actual),
            "y_hat":         float(y_test_predicted),
            "pred_50_lower": float(y_test_predicted) - 0.674 * train_rmse,
            "pred_50_upper": float(y_test_predicted) + 0.674 * train_rmse,
            "pred_80_lower": float(y_test_predicted) - 1.282 * train_rmse,
            "pred_80_upper": float(y_test_predicted) + 1.282 * train_rmse,
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

    
