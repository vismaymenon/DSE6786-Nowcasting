import hdmpy as hd
import numpy as np
import pandas as pd

## Pull filled ragged edge data into file

def monthly_to_quarterly(df):
    df_resampled = df.resample('Q').mean()
    df_resampled.index = df_resampled.index.to_period('M').to_timestamp()
    return df_resampled

## Fit Lasso model based on poos format
def fit_lasso(df_X, gdp):
    X_train = df_X.iloc[:-1]
    y_train = gdp.iloc[:-1].values
    X_test = df_X.iloc[[-1]]
    y_test_actual = float(gdp.iloc[-1])
    model = hd.rlasso(df_X, gdp, post=True)
    coefs = np.nan_to_num(np.array(model.est["coefficients"]).flatten())
    intercept = float(model.est["intercept"])
    y_train_predicted = intercept + X_train.values @ coefs
    y_test_predicted = float(intercept + X_test.values @ coefs)
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "y_train_predicted": y_train_predicted,
        "X_test": X_test,
        "y_test_actual": y_test_actual,
        "y_test_predicted": y_test_predicted
    }



# def predict_lasso(df_X, gdp):
#     X_arr = df_X.values
#     gdp_arr = gdp.values
#     n = len(gdp_arr)
#     n_train_start = n-100
#     forecasts, actuals, indices = [], [], []
#     for t in range(n_train_start, n):
#         start = (t - n_train_start)
#         X_tr, y_tr = X_arr[start:t], gdp_arr[start:t]
#         x_te = X_arr[t]
#         y_hat = fit_lasso(X_tr, y_tr, x_te)
#         forecasts.append(y_hat)
#         actuals.append(gdp_arr[t])
#         indices.append(gdp.index[t])
#         results = pd.DataFrame({ "actual": actuals, "forecast": forecasts, }, index=indices)
#     return results
