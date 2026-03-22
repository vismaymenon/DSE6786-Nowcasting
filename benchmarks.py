import numpy as np
import pandas as pd
from typing import Callable

import load_data   
import poos
import autoregressive

API_KEY = "cd30e8d67ebd36672d4b0ebfc5069427"
series_id = "A191RL1Q225SBEA"  # Real GDP (quarterly, seasonally adjusted, chained 2012 dollars)

# -- Load data 
y_series = load_data.load_transformed_series_latest_release(series_id, API_KEY)

# ── AR(2)

X_df = pd.DataFrame({
    "lag_1": y_series.shift(1),
    "lag_2": y_series.shift(2)
})

df = pd.concat([X_df, y_series], axis=1).dropna()
X_ar = df.iloc[:, :-1].reset_index(drop=True)
y_ar = df.iloc[:,  -1].reset_index(drop=True)

X_out, y_out, rmse, mae = poos.poos_validation(
    method=autoregressive.ar_model_nowcast,
    X=X_ar,
    y=y_ar,
    prop_train=0.9
)

# ── 4. Results ────────────────────────────────────────────────────────────────
print("=== POOS Results (first 10 rows) ===")
print(y_out.head(10))
print(f"\nRMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"OOS observations: {len(y_out)}")

poos.plot_poos_results(y_ar, y_out, title="Autoregressive Model POOS - AR(2)")