from datetime import datetime
import pandas as pd

# Function to convert inputs into model-friendly format
def preprocess_input(origin, destination, departure_date, departure_time, cabin_type):
    departure_datetime = datetime.combine(departure_date, departure_time)
    date_time = pd.to_datetime(departure_datetime)
    year = date_time.year
    month = date_time.month
    week = date_time.week
    hour = date_time.hour
    minute = date_time.minute
    
    input_data = pd.DataFrame({
        "startingAirport": [origin],
        "destinationAirport": [destination],
        "year": [year],
        "month": [month],
        "week": [week],
        "hour": [hour],
        "minute": [minute],
        "cabinCode": [cabin_type.lower()]
    })
    
    return input_data