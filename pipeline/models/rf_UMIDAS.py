"""
rf_UMIDAS.py
============
Random Forest nowcasting model using U-MIDAS features (X3).

Pipeline
--------
  Step 1 — Load filled data and build X3
            Calls build_X3() from output_x.py, which fetches ragged-edge-imputed
            data from Supabase and constructs the U-MIDAS feature matrix:
            each monthly variable becomes 3 quarterly features (_m1/_m2/_m3).

  Step 2 — Fit RF and run POOS
            fit_rf_umidas() is a POOS-compatible wrapper:
              - last row of X / last element of y = test observation
              - all preceding rows = training set
              - max_features = 0.3 (rule-of-thumb default, ~1/3 of features)
              - no cross-validation needed
            poos_validation() from poos.py handles the expanding window loop.
"""

import sys
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# ── Config ────────────────────────────────────────────────────────────────────
N_TREES      = 1000
MAX_FEATURES = 0.3   
RANDOM_STATE = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent   # pipeline/models/
PIPELINE_DIR = THIS_DIR.parent                   # pipeline/
PROJECT_DIR  = PIPELINE_DIR.parent               # project root

sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from output_x import load_filled_data, build_X3
from poos import poos_validation, plot_poos_results


# =============================================================================
# POOS-COMPATIBLE FIT FUNCTION
# =============================================================================

def fit_rf_umidas(df_X: pd.DataFrame, gdp: pd.Series) -> dict:
    """
      Hyperparameters:
      max_features = 0.3   (rule-of-thumb ~1/3; no CV needed)
      max_depth    = None  (fully grown trees)
      n_estimators = 500
    """
    X_train = df_X.iloc[:-1]
    y_train = gdp.iloc[:-1].values
    X_test  = df_X.iloc[[-1]]
    y_test_actual = float(gdp.iloc[-1])

    rf = RandomForestRegressor(
        n_estimators=N_TREES,
        max_features=MAX_FEATURES,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train.values, y_train)

    y_train_predicted = rf.predict(X_train.values)
    y_test_predicted  = float(rf.predict(X_test.values)[0])

    print(f"Predicting quarter: {X_test.index[0]}")

    return {
        "X_train":           X_train,
        "y_train":           y_train,
        "y_train_predicted": y_train_predicted,
        "X_test":            X_test,
        "y_test_actual":     y_test_actual,
        "y_test_predicted":  y_test_predicted,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # ── Step 1: Load data and build X3 (U-MIDAS feature matrix) ──────────────
    print("=" * 55)
    print("Step 1: Load filled data and build X3")
    print("=" * 55)
    df_md, df_qd = load_filled_data()
    X, y = build_X3(df_md, df_qd)

    # ── Step 2: Run POOS over RF U-MIDAS ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("Step 2: POOS evaluation")
    print("=" * 55)
    X_out, y_df, rmse, mae = poos_validation(
        method=fit_rf_umidas,
        X=X,
        y=y,
        num_test=100,
    )

    print(f"\nOut-of-sample RMSE : {rmse:.6f}")
    print(f"Out-of-sample MAE  : {mae:.6f}")
    print(f"OOS observations   : {len(y_df)}")
    print("\nFirst 5 POOS rows:")
    print(y_df.head().to_string())

    plot_poos_results(y, y_df, title="RF U-MIDAS — POOS Forecast vs Actual")