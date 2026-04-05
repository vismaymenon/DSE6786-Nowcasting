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


def randomForest(df_X: pd.DataFrame, gdp: pd.Series) -> dict:
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

