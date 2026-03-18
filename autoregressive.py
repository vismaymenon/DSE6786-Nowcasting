import numpy as np
import pandas as pd
import statsmodels.api as sm


def create_lags(y, p):
    """
    Create lagged dataframe for AR(p)
    """
    df = pd.DataFrame({"y": y})
    for i in range(1, p + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df = df.dropna()
    return df

def ar_model_nowcast(y):
    best_p = 2
    # Step 2: create lagged data
    df = create_lags(y, best_p)

    # Step 3: train-test split (last obs = test)
    train_df = df.iloc[:-1]
    test_df = df.iloc[-1:]

    y_train = train_df["y"]
    X_train = train_df.drop(columns=["y"])
    X_train = sm.add_constant(X_train)

    y_test_actual = test_df["y"].values[0]
    X_test = test_df.drop(columns=["y"])
    X_test = sm.add_constant(X_test)

    # Step 4: fit model
    model = sm.OLS(y_train, X_train).fit()

    # Step 5: predictions
    y_train_predicted = model.predict(X_train)
    y_test_predicted = model.predict(X_test).values[0]

    # Step 6: coefficients + SE
    coef = model.params
    se = model.bse

    coef_df = pd.DataFrame([coef, se])
    coef_df.index = ["coef", "se"]

    return {
        "best_p": best_p,
        "coefficients": coef_df,
        "X": X_train,
        "y_actual": y_train.values,
        "y_train_predicted": y_train_predicted.values,
        "y_test_actual": y_test_actual,
        "y_test_predicted": y_test_predicted
    }