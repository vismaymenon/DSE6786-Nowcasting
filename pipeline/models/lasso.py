import hdmpy as hd
import numpy as np
import pandas as pd

def fit_lasso(df_X: pd.DataFrame, gdp: pd.Series) -> dict:
    # Split into train (all but last) and single test row (last)
    X_train = df_X.iloc[:-1].copy()
    y_train = gdp.iloc[:-1].copy()
    X_test  = df_X.iloc[[-1]].copy()
    y_test_actual = float(gdp.iloc[-1])

    # 1) Drop zero-variance columns
    col_std = X_train.std()
    zero_var_cols = col_std[col_std == 0].index.tolist()
    if zero_var_cols:
        print("Dropping zero-variance columns:", zero_var_cols)
        X_train = X_train.drop(columns=zero_var_cols)
        X_test  = X_test.drop(columns=zero_var_cols, errors="ignore")

    # 2) Drop columns that are *mostly* NaN in the training window
    nan_counts = X_train.isna().sum()
    mostly_nan_cols = nan_counts[nan_counts > 0.9 * len(X_train)].index.tolist()
    if mostly_nan_cols:
        print("Dropping mostly-NaN columns:", mostly_nan_cols)
        X_train = X_train.drop(columns=mostly_nan_cols)
        X_test  = X_test.drop(columns=mostly_nan_cols, errors="ignore")

    # 3) Drop any remaining rows with NaNs in training
    mask_valid_rows = X_train.notna().all(axis=1)
    if not mask_valid_rows.all():
        print(f"Dropping {(~mask_valid_rows).sum()} training rows with NaNs")
        X_train = X_train[mask_valid_rows]
        y_train = y_train[mask_valid_rows]

    if X_train.empty:
        raise ValueError("No non-NaN rows/columns left in X_train after cleaning.")

    # 4) Fit LASSO via hdmpy (no NaNs allowed)
    y_train_arr = y_train.values

    model = hd.rlasso(X_train, y_train_arr, post=True, homoskedastic=False)
    coefs = np.nan_to_num(np.array(model.est["coefficients"]).flatten())
    coefs = coefs[1:]  # drop intercept
    intercept = float(np.array(model.est["intercept"]).flat[0])

    y_train_predicted = intercept + X_train.values @ coefs
    y_test_predicted  = float(intercept + (X_test.values @ coefs)[0])

    print(f"Predicting quarter: {X_test.index[0]}")
    print("LASSO coefficients (non-zero):")
    for col, coef in zip(X_train.columns, coefs):
        if coef != 0:
            print(f"  {col}: {coef:.4f}")

    return {
        "X_train":           X_train,
        "y_train":           y_train_arr,
        "y_train_predicted": y_train_predicted,
        "X_test":            X_test,
        "y_test_actual":     y_test_actual,
        "y_test_predicted":  y_test_predicted,
    }