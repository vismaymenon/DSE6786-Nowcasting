from fetch_functions import fetch_confidence_intervals
from fetch_functions import fetch_historical_data
from fetch_functions import fetch_evaluation_metrics
from fetch_functions import quarter_to_dates
from fetch_functions import fetch_nowcast_data
from fetch_functions import fetch_nowcast_x_labels


data, month_labels = fetch_nowcast_data("2024:Q4")
print("month_labels:", month_labels)
for model, values in data.items():
    print(f"{model}: {values}")

# Test fetch_nowcast_x_labels
labels = fetch_nowcast_x_labels("2024:Q4")
print("x_labels:", labels)

# Test fetch_confidence_intervals
result3 = fetch_confidence_intervals("2024:Q4", "AR_Benchmark")
print("CI month_labels:", result3[0])
print("ci_50_lower:", result3[1])
print("ci_50_upper:", result3[2])