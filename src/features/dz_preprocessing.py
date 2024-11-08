import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz

# Mapping and timezone definitions
cabin_priority = {
    'basic': 0,
    'coach': 1,
    'premium coach': 2,
    'first': 3,
    'business': 4
}

airport_combinations_encoded = {
    'LGA_OAK': 55, 'PHL_OAK': 54, 'EWR_OAK': 53, 'PHL_SFO': 52, 'JFK_OAK': 51,
    'JFK_DEN': 50, 'LGA_SFO': 49, 'JFK_ORD': 48, 'EWR_SFO': 47, 'PHL_LAX': 46,
    'LGA_IAD': 45, 'JFK_SFO': 44, 'PHL_DEN': 43, 'LGA_LAX': 42, 'JFK_LAX': 41,
    'EWR_LAX': 40, 'JFK_IAD': 39, 'EWR_PHL': 38, 'PHL_EWR': 37, 'PHL_IAD': 36,
    'EWR_DEN': 35, 'LGA_PHL': 34, 'EWR_IAD': 33, 'PHL_LGA': 32, 'PHL_DFW': 31,
    'JFK_PHL': 30, 'PHL_DTW': 29, 'JFK_DFW': 28, 'JFK_DTW': 27, 'LGA_DEN': 26,
    'PHL_JFK': 25, 'EWR_DFW': 24, 'PHL_CLT': 23, 'PHL_ORD': 22, 'EWR_DTW': 21,
    'JFK_ATL': 20, 'PHL_ATL': 19, 'PHL_MIA': 18, 'EWR_ORD': 17, 'LGA_DFW': 16,
    'EWR_CLT': 15, 'JFK_CLT': 14, 'EWR_MIA': 13, 'EWR_ATL': 12, 'LGA_DTW': 11,
    'LGA_ATL': 10, 'PHL_BOS': 9, 'LGA_MIA': 8, 'LGA_CLT': 7, 'JFK_MIA': 6,
    'LGA_ORD': 5, 'EWR_LGA': 4, 'JFK_BOS': 3, 'EWR_BOS': 2, 'LGA_BOS': 1
}

eastern_tz = pytz.timezone('America/New_York')

# Function to preprocess user input for model prediction
def preprocess_input(origin, destination, departure_date, departure_time, cabin_type):
    # Calculate `segmentsCabinCode_cleaned`
    segmentsCabinCode_cleaned = cabin_priority.get(cabin_type, 1)

    # Calculate `days_ahead` in Eastern Time
    today = datetime.now(eastern_tz).date()
    if isinstance(departure_date, datetime):
        departure_date_obj = departure_date.date()
    else:
        departure_date_obj = departure_date

    days_ahead = (departure_date_obj - today).days

    # Calculate `day`, `day_of_week`, `week_of_year`
    day = departure_date_obj.day
    day_of_week = departure_date_obj.weekday()  # Monday=0, Sunday=6
    week_of_year = departure_date_obj.isocalendar()[1]

    # Encode `airport_combinations`
    combination_key = f"{origin}_{destination}"
    airport_combinations_code = airport_combinations_encoded.get(combination_key, 0)  # Default to 0 if not found

    # Handle `departure_time` as a `datetime.time` object directly
    if isinstance(departure_time, time):  # Using the correct `time` class
        departure_time_in_hours = departure_time.hour + departure_time.minute / 60.0
    else:
        departure_time_obj = datetime.strptime(departure_time, "%H:%M").replace(tzinfo=eastern_tz)
        departure_time_in_hours = departure_time_obj.hour + departure_time_obj.minute / 60.0

    # Calculate `departure_time_sin` and `departure_time_cos`
    departure_time_sin = np.sin(2 * np.pi * departure_time_in_hours / 24)
    departure_time_cos = np.cos(2 * np.pi * departure_time_in_hours / 24)

    # Create a DataFrame with processed features
    input_data = pd.DataFrame([{
        "segmentsCabinCode_cleaned": segmentsCabinCode_cleaned,
        "days_ahead": days_ahead,
        "day": day,
        "day_of_week": day_of_week,
        "week_of_year": week_of_year,
        "airport_combinations_encoded": airport_combinations_code,
        "departure_time_sin": departure_time_sin,
        "departure_time_cos": departure_time_cos,
    }])

    return input_data
