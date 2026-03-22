import numpy as np
import pandas as pd
import statsmodels.api as sm


def ar_model_nowcast(X, y):
    X_train = X.iloc[:-1].copy()
    y_train = y.iloc[:-1].values
    X_test  = X.iloc[[-1]].copy()
    y_test_actual = float(y.iloc[-1])

    # Add constant for intercept
    X_train = sm.add_constant(X_train, has_constant='add')
    X_test  = sm.add_constant(X_test,  has_constant='add')

    model = sm.OLS(y_train, X_train).fit()

    y_train_predicted = model.predict(X_train)
    y_test_predicted  = model.predict(X_test).values[0]

    return {
        "X_train":           X_train,
        "y_train":           y_train,
        "y_train_predicted": y_train_predicted,
        "X_test":            X_test,
        "y_test_actual":     y_test_actual,
        "y_test_predicted":  y_test_predicted,
    }