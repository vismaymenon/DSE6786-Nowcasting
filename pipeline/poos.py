import numpy as np
import pandas as pd
from typing import Callable
from datetime import date
import os
from dotenv import load_dotenv

# ── Benchmark model ───────────────────────────────────────────────────────────

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

# ── POOS ──────────────────────────────────────────────────────────────────────

def poos_validation(
    method: Callable,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    num_test: int = 100,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:

    X = pd.DataFrame(X)  # ← remove reset_index
    y = pd.Series(y)     # ← remove reset_index
    n = len(y)

    train_size = n - num_test
    test_indices, actuals = [], []
    preds_point, preds_50_lower, preds_50_upper, preds_80_lower, preds_80_upper = [], [], [], [], []

    for t in range(n - train_size):
        X_window = X.iloc[t:t+train_size+1]
        y_window = y.iloc[t:t+train_size+1]

        _, y_train_actual, y_train_predicted, _, y_test_actual, y_test_predicted = method(X_window, y_window).values()
        std_error = np.std(y_train_actual - y_train_predicted)

        test_indices.append(y.index[t + train_size])  # ← use datetime index
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

    return X.iloc[range(n - train_size)].copy(), y_df, rmse, mae


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
    plt.show()

# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv() 
    API_KEY = os.getenv("API_KEY")

    # Load & transform a FRED series as target (y)
    # Using INDPRO (Industrial Production) as an example
    y_series = load_data.load_transformed_series_latest_release("INDPRO", API_KEY)

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

    
