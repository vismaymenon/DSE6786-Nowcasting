from supabase_client import supabase



######## Function 1: Getting Nowcast Data ########

#### Helper Function ####
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

#### Driver Function ####
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