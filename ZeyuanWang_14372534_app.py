import streamlit as st
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import joblib
from datetime import datetime

# List of available airports and associated airport codes
airport_codes = {
    "John F. Kennedy International Airport (New York City, NY)": "JFK",
    "Newark Liberty International Airport (Newark, NJ)": "EWR",
    "LaGuardia Airport (New York City, NY)": "LGA",
    "Philadelphia International Airport (Philadelphia, PA)": "PHL",
    "Hartsfield-Jackson Atlanta International Airport (Atlanta, GA)": "ATL",
    "Miami International Airport (Miami, FL)": "MIA",
    "Charlotte Douglas International Airport (Charlotte, NC)": "CLT",
    "Dallas/Fort Worth International Airport (Dallas, TX)": "DFW",
    "Los Angeles International Airport (Los Angeles, CA)": "LAX",
    "San Francisco International Airport (San Francisco, CA)": "SFO",
    "Oakland International Airport (Oakland, CA)": "OAK",
    "Denver International Airport (Denver, CO)": "DEN",
    "O'Hare International Airport (Chicago, IL)": "ORD",
    "Logan International Airport (Boston, MA)": "BOS",
    "Washington Dulles International Airport (Washington, D.C.)": "IAD",
    "Detroit Metropolitan Wayne County Airport (Detroit, MI)": "DTW"
}

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

# Function to preprocess input for Zeyaun Wang's model
def preprocess_input_for_zeyuan(origin, destination, departure_date, departure_time, cabin_type, label_encoders, feature_scaler):
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


# Navigation system (navbar using radio button)
st.sidebar.title("FlyFare")
page = st.sidebar.radio("Go to", ["Home", "Airfare Prediction", "About", "Repository"])

# Display the content of each page based on the user's selection
if page == "Home":
    st.title("Welcome to the FlyFare: Your Airfare Estimator")
    st.write("This application helps you estimate airfare based on your trip details. Use the 'Airfare Prediction' section to get started.")
    st.image("https://i.pinimg.com/736x/23/fd/cb/23fdcb0e65df24eb742b0fccbbddc65b.jpg", caption="Welcome to FlyFare!", use_column_width=True)

elif page == "Airfare Prediction":
    st.title("FlyFare")
    st.write("Estimate your airfare by providing details of your trip.")
    
    # Select origin and destination airports
    origin_airport = st.selectbox("Select Origin Airport", list(airport_codes.keys()))
    
    # Filter the list of destination airports to exclude the selected origin airport
    destination_airports = [airport for airport in airport_codes.keys() if airport != origin_airport]
    destination_airport = st.selectbox("Select Destination Airport", destination_airports)
    
    # Collect other user inputs
    departure_date = st.date_input("Departure Date")
    departure_time = st.time_input("Departure Time")
    cabin_type = st.selectbox("Cabin Type", ["coach", "premium coach", "business", "first"])
    
    # Button to trigger the prediction
    if st.button("Predict Fare"):
        # Convert selected airports to IATA codes
        origin_code = airport_codes[origin_airport]
        destination_code = airport_codes[destination_airport]
        
        # Load Zeyaun Wang's LSTM model if the selected airport is managed by him
        if origin_code in ["ATL", "MIA", "CLT", "DFW"]:
            model, label_encoders, feature_scaler, target_scaler = load_zeyuan_model()
            input_data_scaled = preprocess_input_for_zeyuan(origin_code, destination_code, departure_date, departure_time, cabin_type, label_encoders, feature_scaler)
            
            # Prepare input sequence
            sequence_length = 10
            input_sequence = np.repeat(input_data_scaled, sequence_length, axis=0)
            input_sequence = np.expand_dims(input_sequence, axis=0)
            
            # Make prediction
            predicted_fare_scaled = model.predict(input_sequence)[0][0]
            predicted_fare = target_scaler.inverse_transform([[predicted_fare_scaled]])[0][0]
            
            # Display the result
            if predicted_fare < 0 or predicted_fare > 5000:  
                st.write("There are currently no suitable tickets")
            else:
                st.success(f"Predicted Fare: ${predicted_fare:.2f}")
        else:
            st.error("No models available for the selected origin.")
            
elif page == "About":
    st.title("About FlyFare")
    st.write("""
        FlyFare is developed as a collaborative machine learning project where multiple models are trained 
        to predict airfare based on various factors like origin, destination, departure date, and cabin class.
        Each student in the project is responsible for managing a specific set of airports. Below is the breakdown:
    """)
    
    st.subheader("Student Contributions")
    st.write("""
    **Dezhou Zhang**: Managed airports in the East Coast United States:
    - JFK: John F. Kennedy International Airport (New York City, NY)
    - EWR: Newark Liberty International Airport (Newark, NJ)
    - LGA: LaGuardia Airport (New York City, NY)
    - PHL: Philadelphia International Airport (Philadelphia, PA)
    
    **Zeyaun Wang**: Managed airports in the Southern United States:
    - ATL: Hartsfield-Jackson Atlanta International Airport (Atlanta, GA)
    - MIA: Miami International Airport (Miami, FL)
    - CLT: Charlotte Douglas International Airport (Charlotte, NC)
    - DFW: Dallas/Fort Worth International Airport (Dallas, TX)
    
    **Hoa Deng**: Managed airports in the Western United States:
    - LAX: Los Angeles International Airport (Los Angeles, CA)
    - SFO: San Francisco International Airport (San Francisco, CA)
    - OAK: Oakland International Airport (Oakland, CA)
    - DEN: Denver International Airport (Denver, CO)
    
    **Rusan Vaidya**: Managed airports in the Central and Major Hubs:
    - ORD: O'Hare International Airport (Chicago, IL)
    - BOS: Logan International Airport (Boston, MA)
    - IAD: Washington Dulles International Airport (Washington, D.C.)
    - DTW: Detroit Metropolitan Wayne County Airport (Detroit, MI) 
    """)

    st.write("""
    Each student was responsible for gathering historical airfare data, training machine learning models,
    and integrating their model into the FlyFare app. This collaborative approach allows for multiple models 
    to be compared and validated for better accuracy.
    """)

elif page == "Repository":
    st.title("Project Repo")
    st.write("""
        For more details, check out the github repo:
        - Github Experiment Repo: https://github.com/rusanvaidya/at3-travel-airfare-group3
        - Github Streamlit Repo: https://github.com/rusanvaidya/at3-grp3-airfare-streamlit-app
    """)
