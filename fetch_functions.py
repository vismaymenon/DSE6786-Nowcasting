from supabase_client import supabase

######## Helper Function ########
## Converts quarter string to start and end dates ##
# Needed 'cuz we don't have a column for quarter #
def quarter_to_dates(quarter: str):

    year, q = quarter.split(":")
    quarter_map = {
        "Q1": ("01-01", "03-31"),
        "Q2": ("04-01", "06-30"),
        "Q3": ("07-01", "09-30"),
        "Q4": ("10-01", "12-31"),
    }
    start, end = quarter_map[q]
    return f"{year}-{start}", f"{year}-{end}"

######## Function 1: Getting Nowcast Data ########

def fetch_nowcast_data(quarter: str) -> dict[str, list[float]]:
    
    # Step 0: Get input's (quarter) start and end dates:
    quarter_start, quarter_end = quarter_to_dates(quarter)
    
    # Step 1: Get rows from model_forecasts table for the specified quarter
    result = supabase.table("model_forecasts") \
        .select("*") \
        .gte("month_date", quarter_start) \
        .lte("month_date", quarter_end) \
        .order("month_date") \
        .execute()
        
    forecasts = result.data
    
    # Step 2: Reshape into desired format
    data = {}
    for row in forecasts:
        model = row["model_name"]
        if model not in data:
            data[model] = []
        data[model].append(row["nowcast"])
    
    # Step 3: Get the month labels
    models_requested = list(data.keys())
    
    month_labels = [
        row["month_date"] for row in forecasts
        if row["model_name"] == models_requested[0]
    ]
    
    return data, month_labels


######## Function 2: Getting Labels ########
#### This is just a segment of Function 1. ####
def fetch_nowcast_x_labels(quarter: str) -> list[str]:
    
    # Step 0: Get input's (quarter) start and end dates:
    quarter_start, quarter_end = quarter_to_dates(quarter)
    
    # Step 1: Repeat step 1; query supabase
    result = supabase.table("model_forecasts") \
    .select("*") \
    .gte("month_date", quarter_start) \
    .lte("month_date", quarter_end) \
    .order("month_date") \
    .execute()
        
    forecasts = result.data

    # Step 2: Pick just one model and get all its dates, given that all models have the same dates
    ref_model = forecasts[0]["model_name"]
    month_labels = [
        row ["month_date"] for row in forecasts
        if row["model_name"] == ref_model
    ]
    
    return month_labels

######## Function 3: Fetch Confidence Intervals ########

def fetch_confidence_intervals(quarter: str, model: str) -> tuple[list[str], list[float], list[float]]:
    
    # Step 0: Get input's (quarter) start and end dates:
    quarter_start, quarter_end = quarter_to_dates(quarter)
    
    # Step 1: Repeat step 1; query supabase
    result = supabase.table("model_forecasts") \
    .select("month_date", "ci_50_lb", "ci_50_ub", "ci_80_lb", "ci_80_ub") \
    .eq("model_name", model) \
    .gte("month_date", quarter_start) \
    .lte("month_date", quarter_end) \
    .order("month_date") \
    .execute()
        
    forecasts = result.data

    # Step 2: Extract each column into its own list
    month_labels = [row["month_date"] for row in forecasts]
    ci_50_lower = [row["ci_50_lb"] for row in forecasts]
    ci_50_upper = [row["ci_50_ub"] for row in forecasts]
    ci_80_lower = [row["ci_80_lb"] for row in forecasts]
    ci_80_upper = [row["ci_80_ub"] for row in forecasts]
    
    return month_labels, ci_50_lower, ci_50_upper, ci_80_lower, ci_80_upper


######## Function 4: Fetch Historical Data ########
def fetch_historical_data(start_date, end_date, flash_month: int) -> tuple[list[str], list[float], dict[str, list[float]]]:
    
    # Step 1: Get actual GDP values from gdp table
    actuals_results = supabase.table("gdp") \
        .select("sasdate", "GDPC1_t") \
        .gte("sasdate", str(start_date)) \
        .lte("sasdate", str(end_date)) \
        .order("sasdate") \
        .execute()
    
    actuals_rows = actuals_results.data
    
    # Step 2: Get model predictions 
    preds_results = supabase.table("model_forecasts") \
        .select("month_date, model_name, nowcast") \
        .gte("month_date", str(start_date))
        .lte("month_date", str(end_date)) \
        .order("month_date") \
        .execute()
    
    preds_rows = preds_results.data
    
    # Step 3: Extract actuals into lists
    quarter_labels = [row["sasdate"]  for row in actuals_rows]
    actual_values  = [row["GDPC1_t"] for row in actuals_rows]

    # Step 4: Reshape predictions by model
    predictions = {}
    for row in preds_rows:
        model = row["model_name"]
        if model not in predictions:
            predictions[model] = []
        predictions[model].append(row["nowcast"])

    return quarter_labels, actual_values, predictions

######## Function 5: Fetch Evaluation Metrics (RMSE) ########
## The following function fetches RMSE values for a single mode ##
def fetch_evaluation_metrics(model: str, version: int) -> dict:
    
    # Step 1: Query Supabase for all rows matching requested models
    result = supabase.table("rmse") \
        .select("model", "version", "rmse") \
        .eq("model", model) \
        .eq("version", version) \
        .execute()
    
    row = result.data[0]
    return {"model": row["model"], "version": row["version"], "rmse": row["rmse"]}
