import hdmpy as hd
import numpy as np
import pandas as pd


def monthly_to_quarterly(df):
    df_resampled = df.resample('Q', on='sasdate').mean()
    df_resampled.index = df_resampled.index.to_period('M').to_timestamp()
    return df_resampled

## Fit Lasso model based on poos format
def fit_lasso(df_X, gdp):
    X_train = df_X.iloc[:-1]
    y_train = gdp.iloc[:-1].values
    X_test = df_X.iloc[[-1]]
    y_test_actual = float(gdp.iloc[-1])

    # print("train shape:", X_train.shape)

    model = hd.rlasso(X_train, y_train, post=True)
    coefs = np.nan_to_num(np.array(model.est["coefficients"]).flatten())
    coefs = coefs[1:] # drop intercept
    # print("coefficients:",coefs.shape)
    # print("train shape:", X_train.shape)
    intercept = float(model.est["intercept"])
    y_train_predicted = intercept + X_train.values @ coefs
    y_test_predicted = float(intercept + X_test.values @ coefs)

    print(f"Predicting quarter: {X_test.index[0]}")
    print("LASSO coefficients (non-zero):")
    for col, coef in zip(X_train.columns, coefs):
        if coef != 0:
            print(f"  {col}: {coef:.4f}")
    
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


## TEST ##
# import sys, os
# # Add the project root to the path for relative imports
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from dotenv import load_dotenv
# load_dotenv()

# from pipeline import poos
# from pipeline import models
# from pipeline.models import autoregressive
# from pipeline.models import rf_benchmark
# from pipeline.models import rf_UMIDAS as rf_umidas_module
# from pipeline.models import rf_avg as rf_avg_module

# import pandas as pd
# from pathlib import Path

# NUM_TEST   = 100
# AR_LAGS    = 2
# RF_LAGS    = 4

# DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# def build_lag_features(gdp, n_lags):
#     df = gdp.rename("gdp_growth").to_frame()
#     for lag in range(1, n_lags + 1):
#         df[f"lag_{lag}"] = df["gdp_growth"].shift(lag)
#     df.dropna(inplace=True)
#     X = df[[f"lag_{i}" for i in range(1, n_lags + 1)]]
#     y = df["gdp_growth"]
#     return X, y


# # ── Load data ─────────────────────────────────────────────────────────────────
# gdp = pd.read_csv(DATA_DIR / "gdp.csv", parse_dates=["sasdate"])
# gdp = gdp.set_index("sasdate").sort_index().squeeze()
# gdp = gdp[gdp.index.notna()]

# ## Pull filled ragged edge data into file

# ── RF LASSO POOS ───────────────────────────────────────────────────────────
# print("\n=== LASSO ===")
# df_md_filled, df_qd_filled = rf_umidas_module.load_filled_data()
# X_lasso= monthly_to_quarterly(df_md_filled)
# y_lasso = gdp
# _, lasso_out, lasso_rmse, lasso_mae = poos.poos_validation(
#     method= fit_lasso,
#     X=X_lasso,
#     y=y_lasso,
#     num_test=NUM_TEST,
# )

# print(f"{'LASSO':<25} {lasso_rmse:>8.4f} {lasso_mae:>8.4f}")
# poos.plot_poos_results(y_lasso, lasso_out,  title="LASSO — POOS")