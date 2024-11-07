import streamlit as st
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle


model = tf.keras.models.load_model('model.h5')
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('feature_scaler.pkl', 'rb') as f:
    feature_scaler = pickle.load(f)
with open('target_scaler.pkl', 'rb') as f:
    target_scaler = pickle.load(f)

st.title("Airfare Prediction App")


sequence_length = 10  
st.sidebar.header("Input Your Flight Details")
origin_airport = st.sidebar.selectbox("Origin Airport", options=label_encoders['startingAirport'].classes_)
destination_airport = st.sidebar.selectbox("Destination Airport", options=label_encoders['destinationAirport'].classes_)
departure_date = st.sidebar.date_input("Departure Date", min_value=datetime.date.today())
departure_time = st.sidebar.time_input("Departure Time", value=datetime.time(8, 0))
cabin_type = st.sidebar.selectbox("Cabin Type", options=label_encoders['segmentsCabinCode'].classes_)

if st.sidebar.button("Predict Fare"):
    input_data = pd.DataFrame({
        'startingAirport': [origin_airport],
        'destinationAirport': [destination_airport],
        'flightDate': [departure_date],
        'segmentsDepartureTimeRaw': [datetime.datetime.combine(departure_date, departure_time)],
        'segmentsCabinCode': [cabin_type]
    })

    input_data['startingAirport'] = label_encoders['startingAirport'].transform(input_data['startingAirport'])
    input_data['destinationAirport'] = label_encoders['destinationAirport'].transform(input_data['destinationAirport'])
    input_data['segmentsCabinCode'] = label_encoders['segmentsCabinCode'].transform(input_data['segmentsCabinCode'])

    epoch = datetime.datetime(1970, 1, 1)
    input_data['flightDate'] = input_data['flightDate'].apply(lambda x: (datetime.datetime.combine(x, datetime.time.min) - epoch).total_seconds() / 1e5)
    input_data['segmentsDepartureTimeRaw'] = input_data['segmentsDepartureTimeRaw'].apply(lambda x: (x - epoch).total_seconds() / 1e5)

    input_data_scaled = feature_scaler.transform(input_data)

    input_sequence = np.repeat(input_data_scaled, sequence_length, axis=0)
    input_sequence = np.expand_dims(input_sequence, axis=0) 

    predicted_fare_scaled = model.predict(input_sequence)[0][0]

    predicted_fare = target_scaler.inverse_transform([[predicted_fare_scaled]])[0][0]

    if predicted_fare < 0 or predicted_fare > 5000:  
        st.write("There are currently no suitable tickets")
    else:
        st.write(f"Predicted Fare: ${predicted_fare:.2f}")
