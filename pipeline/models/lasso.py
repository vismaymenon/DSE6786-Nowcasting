import hdmpy as hd
import numpy as np
import pandas as pd


def fit_lasso(df_X, gdp):
    X_train = df_X.iloc[:-1]
    y_train = gdp.iloc[:-1].values
    X_test  = df_X.iloc[[-1]]
    y_test_actual = float(gdp.iloc[-1])

    # ── Drop zero-variance columns ────────────────────────────────────────────
    col_std = X_train.std()
    zero_var_cols = col_std[col_std == 0].index.tolist()
    if zero_var_cols:
        X_train = X_train.drop(columns=zero_var_cols)
        X_test  = X_test.drop(columns=zero_var_cols)

    model = hd.rlasso(X_train, y_train, post=False)
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
        "y_train":           y_train,
        "y_train_predicted": y_train_predicted,
        "X_test":            X_test,
        "y_test_actual":     y_test_actual,
        "y_test_predicted":  y_test_predicted,
    }