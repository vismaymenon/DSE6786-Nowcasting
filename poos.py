# poos.py

import numpy as np
import pandas as pd
from typing import Callable
import load_data

# ── Benchmark model ───────────────────────────────────────────────────────────

def placeholder_model(X, y):
    """
    Benchmark (unconditional mean) model.
    Treats the last row of X and last element of y as the test observation.

    Inputs:
        X : pd.DataFrame, shape (t+1, n_features)  — last row is test
        y : pd.Series,    shape (t+1,)              — last element is test

    Outputs:
        coefficients     (pd.DataFrame) : 1-row df of NaNs (no real coeffs)
        X                (pd.DataFrame) : training X
        y_actual         (np.ndarray)   : training y
        y_train_predicted(np.ndarray)   : in-sample predictions (all = mean)
        y_test_actual    (float)        : held-out true value
        y_test_predicted (float)        : prediction = mean of training y
    """
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1].values
    y_test_actual = float(y.iloc[-1])

    train_mean = float(np.mean(y_train))

    coefficients = pd.DataFrame(
        [[np.nan] * X_train.shape[1]],
        columns=X_train.columns if hasattr(X_train, "columns") else range(X_train.shape[1])
    )

    y_train_predicted = np.full_like(y_train, fill_value=train_mean, dtype=float)
    y_test_predicted  = train_mean

    return coefficients, X_train, y_train, y_train_predicted, y_test_actual, y_test_predicted


# ── POOS ──────────────────────────────────────────────────────────────────────

def poos_validation(
    method: Callable,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    min_train_size: float | int = 0.9,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Pseudo Out-of-Sample (POOS) expanding-window validation.

    Parameters
    ----------
    method         : Callable  — signature: method(X, y) -> (coefficients, X, y_actual,
                                  y_train_predicted, y_test_actual, y_test_predicted)
    X              : feature matrix (n_samples, n_features)
    y              : target array   (n_samples,)
    min_train_size : if float in (0,1), treated as proportion of sample;
                     if int >= 2, treated as absolute number of training obs.
                     Defaults to 0.9 (90% burn-in).
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    n = len(y)

    # Resolve min_train_size to an integer index
    if isinstance(min_train_size, float) and 0 < min_train_size < 1:
        min_train_size = max(int(n * min_train_size), 2)
    else:
        min_train_size = int(min_train_size)

    if min_train_size >= n:
        raise ValueError(f"min_train_size ({min_train_size}) must be < n ({n}).")

    predictions, actuals, test_indices, all_coefficients = [], [], [], []

    for t in range(min_train_size, n):
        X_window = X.iloc[: t + 1]   # rows 0..t  (last row = test)
        y_window = y.iloc[: t + 1]

        coefficients, _, _, _, y_test_actual, y_test_predicted = method(X_window, y_window)

        predictions.append(float(y_test_predicted))
        actuals.append(float(y_test_actual))
        test_indices.append(t)
        all_coefficients.append(coefficients.iloc[0])

    predictions = np.array(predictions)
    actuals     = np.array(actuals)
    errors      = actuals - predictions

    y_actual_df = pd.DataFrame(
        {
            "y_hat":  predictions,
            "y_-40":  predictions + np.quantile(errors, 0.10),
            "y_-25":  predictions + np.quantile(errors, 0.25),
            "y_25":   predictions + np.quantile(errors, 0.75),
            "y_40":   predictions + np.quantile(errors, 0.90),
            "y_true": actuals,
        },
        index=test_indices,
    )

    return X.iloc[test_indices].copy(), predictions, y_actual_df


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    API_KEY = "cd30e8d67ebd36672d4b0ebfc5069427"

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
    X = df.iloc[:, :-1].reset_index(drop=True)
    y = df.iloc[:,  -1].reset_index(drop=True)

    print(f"Sample size: {len(y)}")
    print(f"Features:    {X.columns.tolist()}\n")

    # Run POOS with benchmark model
    X_out, y_pred, y_actual_df = poos_validation(
        method=placeholder_model,
        X=X,
        y=y,
        min_train_size=0.9,
    )

    print("=== POOS Results (first 5 rows) ===")
    print(y_actual_df.head())

    rmse = np.sqrt(np.mean((y_actual_df["y_true"] - y_actual_df["y_hat"]) ** 2))
    mae  = np.mean(np.abs(y_actual_df["y_true"] - y_actual_df["y_hat"]))
    print(f"\nOut-of-sample RMSE : {rmse:.6f}")
    print(f"Out-of-sample MAE  : {mae:.6f}")
    print(f"\nOOS observations   : {len(y_actual_df)}")