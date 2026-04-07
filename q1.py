from pipeline.pipe import run
import pandas as pd


start = "2026-01-01"
end = "2026-03-31"

# Generate month-end dates
month_ends = pd.date_range(start=start, end=end, freq='M')

# Iterate through them
for date in month_ends:
    run(run_date=date.strftime("%Y-%m-%d"))