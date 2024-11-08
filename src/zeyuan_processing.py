import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import joblib
from datetime import datetime
import tensorflow as tf

# Function to preprocess input for Zeyaun Wang's model
def preprocess_input(origin, destination, departure_date, departure_time, cabin_type, label_encoders, feature_scaler):
    input_data = pd.DataFrame({
        'startingAirport': [origin],
        'destinationAirport': [destination],
        'flightDate': [departure_date],
        'segmentsDepartureTimeRaw': [datetime.combine(departure_date, departure_time)],  # Make sure departure_time is of type datetime.time
        'segmentsCabinCode': [cabin_type]
    })
    
    # Transform categorical columns using label encoders
    input_data['startingAirport'] = label_encoders['startingAirport'].transform(input_data['startingAirport'])
    input_data['destinationAirport'] = label_encoders['destinationAirport'].transform(input_data['destinationAirport'])
    input_data['segmentsCabinCode'] = label_encoders['segmentsCabinCode'].transform(input_data['segmentsCabinCode'])
    
    # Convert date columns to timestamps
    epoch = datetime(1970, 1, 1)
    input_data['flightDate'] = input_data['flightDate'].apply(lambda x: (datetime.combine(x, datetime.min.time()) - epoch).total_seconds() / 1e5)
    input_data['segmentsDepartureTimeRaw'] = input_data['segmentsDepartureTimeRaw'].apply(lambda x: (x - epoch).total_seconds() / 1e5)
    
    # Scale features
    input_data_scaled = feature_scaler.transform(input_data)
    return input_data_scaled
# Mapping function to convert cabin types to the required format
def map_cabin_type(user_input):
    cabin_type_map = {
        "Coach": "coach",
        "Premium": "premium coach",
        "Business": "business",
        "First Class": "first"
    }
    return cabin_type_map.get(user_input, user_input)  # Default to input if not found

# Load Zeyaun Wang's LSTM model and relevant encoders/scalers
def load_zeyuan_model():
    model = tf.keras.models.load_model('model_ZeyuanWang_14372534.h5')
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('feature_scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    with open('target_scaler.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    return model, label_encoders, feature_scaler, target_scaler