"""
lasso_lags.py
=============
LASSO nowcasting model using simple average + lags features (X2).

Pipeline
--------
  Step 1 — Load filled data and build X2
            Calls build_X2() from output_x.py, which constructs quarterly
            averages of monthly variables plus 4 quarterly lags.

  Step 2 — Fit LASSO and run POOS
            fit_lasso() is a POOS-compatible wrapper using hdmpy rlasso.
            poos_validation() from poos.py handles the expanding window loop.
"""

import sys
import hdmpy as hd
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent   # pipeline/models/
PIPELINE_DIR = THIS_DIR.parent                   # pipeline/
PROJECT_DIR  = PIPELINE_DIR.parent               # project root

sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from output_x import load_filled_data, build_X2
from poos import poos_validation, plot_poos_results


# =============================================================================
# POOS-COMPATIBLE FIT FUNCTION
# =============================================================================

def fit_lasso(df_X, gdp):
    X_train = df_X.iloc[:-1]
    y_train = gdp.iloc[:-1].values
    X_test = df_X.iloc[[-1]]
    y_test_actual = float(gdp.iloc[-1])

    model = hd.rlasso(X_train, y_train, post=True)
    coefs = np.nan_to_num(np.array(model.est["coefficients"]).flatten())
    coefs = coefs[1:]  # drop intercept
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


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # ── Step 1: Load data and build X2 (simple average + lags) ───────────────
    print("=" * 55)
    print("Step 1: Load filled data and build X2")
    print("=" * 55)
    df_md, df_qd = load_filled_data()
    X, y = build_X2(df_md, df_qd)

    # Drop the nowcast row (y=NaN) before passing to POOS
    X_eval = X[y.notna()]
    y_eval = y[y.notna()]

    # ── Step 2: Run POOS over LASSO ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Step 2: POOS evaluation")
    print("=" * 55)
    X_out, y_df, rmse, mae = poos_validation(
        method=fit_lasso,
        X=X_eval,
        y=y_eval,
        num_test=100,
    )

    print(f"\nOut-of-sample RMSE : {rmse:.6f}")
    print(f"Out-of-sample MAE  : {mae:.6f}")
    print(f"OOS observations   : {len(y_df)}")
    print("\nFirst 5 POOS rows:")
    print(y_df.head().to_string())

    plot_poos_results(y_eval, y_df, title="LASSO (avg + lags) — POOS Forecast vs Actual")
