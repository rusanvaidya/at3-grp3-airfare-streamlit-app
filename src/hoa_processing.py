import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Instantiate label encoders (use the same encoder as during training)
label_encoder_starting = LabelEncoder()
label_encoder_destination = LabelEncoder()

# Function to preprocess input for prediction
def preprocess_input(origin, destination, departure_date, departure_time, cabin_type):
    # Combine departure date and time into a datetime object
    departure_datetime = datetime.combine(departure_date, departure_time)

    # Extract relevant date features
    departure_hour = departure_datetime.hour
    departure_dayofweek = departure_datetime.weekday()
    departure_month = departure_datetime.month

    # fill default num to run the model
    isBasicEconomy = 0
    isRefundable = 0
    isNonStop = 1
    travelDuration = 120
    totalTravelDistance = 500
    flight_duration = travelDuration

    # Create dummy variables for 'segmentsCabinCode'
    segmentsCabinCode_coach = 1 if cabin_type.lower() == 'coach' else 0
    segmentsCabinCode_first = 1 if cabin_type.lower() == 'first' else 0
    segmentsCabinCode_premium_coach = 1 if cabin_type.lower() == 'premium coach' else 0

    # Label encode the 'startingAirport' and 'destinationAirport'
    origin_encoded = label_encoder_starting.fit_transform([origin])[0]
    destination_encoded = label_encoder_destination.fit_transform([destination])[0]

    # Create a DataFrame for input data
    input_data = pd.DataFrame([{
        "startingAirport": origin_encoded,
        "destinationAirport": destination_encoded,
        "isBasicEconomy": isBasicEconomy,
        "isRefundable": isRefundable,
        "isNonStop": isNonStop,
        "travelDuration": travelDuration,
        "totalTravelDistance": totalTravelDistance,
        "departure_hour": departure_hour,
        "departure_dayofweek": departure_dayofweek,
        "departure_month": departure_month,
        "flight_duration": flight_duration,
        "segmentsCabinCode_coach": segmentsCabinCode_coach,
        "segmentsCabinCode_first": segmentsCabinCode_first,
        "segmentsCabinCode_premium coach": segmentsCabinCode_premium_coach
    }])

    return input_data
