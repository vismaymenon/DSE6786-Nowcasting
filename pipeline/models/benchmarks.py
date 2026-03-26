import numpy as np
import pandas as pd
from typing import Callable
import os
from dotenv import load_dotenv

import pipeline.load_data as load_data   
import pipeline.poos as poos
import pipeline.models.autoregressive as autoregressive

load_dotenv()
API_KEY = os.getenv("API_KEY")
series_id = "A191RL1Q225SBEA"  # Real GDP (quarterly, seasonally adjusted, chained 2012 dollars)

# -- Load data 
y_series = load_data.load_transformed_series_latest_release(series_id=series_id, API_KEY=API_KEY)

# ── AR(p=2, h=2)

X_df = pd.DataFrame({
    "lag_1": y_series.shift(2),
    "lag_2": y_series.shift(3),
    "lag_3": y_series.shift(4),

})

df = pd.concat([X_df, y_series], axis=1).dropna()
X_ar = df.iloc[:, :-1]
y_ar = df.iloc[:,  -1]

X_out, y_out, rmse, mae = poos.poos_validation(
    method=autoregressive.ar_model_nowcast,
    X=X_ar,
    y=y_ar,
    num_test=100
)

# ── 4. Results ────────────────────────────────────────────────────────────────
print("=== POOS Results (first 10 rows) ===")
print(y_out.head(10))
print(f"\nRMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"OOS observations: {len(y_out)}")

poos.plot_poos_results(y_ar, y_out, title="Autoregressive Model POOS - AR(2)")
