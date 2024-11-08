import streamlit as st
import pandas as pd
import joblib
import pickle
import h5
from datetime import datetime
import tensorflow as tf

# Import preprocessing functions with unique names
from src.dz_processing import preprocess_input as preprocess_dz
from src.hoa_processing import preprocess_input as preprocess_hoa
from src.rusan_processing import preprocess_input as preprocess_rusan
from src.zeyuan_processing import preprocess_input as preprocess_zeyuan

# List of available airports
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

# Function to load model based on origin airport
def load_model_and_preprocessor(origin_code):
    if origin_code in ["JFK", "EWR", "LGA", "PHL"]:
        # Load models handled by Dezhou
        model_path = "models/Dezhou_finalXGB.joblib"
        model = joblib.load(model_path)
        preprocess_function = preprocess_dz

    elif origin_code in ["ATL", "MIA", "CLT", "DFW"]:
        # Load models handled by Zeyuan
        model_path = "models/model_ZeyuanWang_14372534.h5"
        model = tf.keras.models.load_model(model_path)
        preprocess_function = preprocess_zeyuan

    elif origin_code in ["LAX", "SFO", "OAK", "DEN"]:
        # Load models handled by Hao
        model_path = "models/model_West_Coast_Hubs.pkl"
        model = joblib.load(model_path)
        preprocess_function = preprocess_hoa

    elif origin_code in ["ORD", "BOS", "IAD", "DTW"]:
        # Load models handled by Rusan
        model_path = "models/model_rusanvaidya_24886400.joblib"
        model = joblib.load(model_path)
        preprocess_function = preprocess_rusan

    else:
        model = None
        preprocess_function = None
        model_path = None

    return model, model_path, preprocess_function

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
    cabin_type = st.selectbox("Cabin Type", ["Coach", "Premium", "Business", "First Class"])
    
    # Button to trigger the prediction
    if st.button("Predict Fare"):
        # Convert selected airports to IATA codes
        origin_code = airport_codes[origin_airport]
        destination_code = airport_codes[destination_airport]
        
        # Load the appropriate model and preprocessor based on the selected origin airport
        model, model_path, preprocess_function = load_model_and_preprocessor(origin_code)
        
        if model is None or preprocess_function is None:
            st.error("No models available for the selected origin.")
        else:
            # Preprocess the input data
            st.success(f"Model loaded from {model_path}")

            input_data = preprocess_function(origin_code, destination_code, departure_date, departure_time, cabin_type)
            
            # Get predictions from the model
            result = model.predict(input_data)
            
            # Display the results
            st.success(f"Predicted Fare: ${result[0]:.2f}")
            
elif page == "About":
    st.title("About FlyFare")
    st.write("""
        FlyFare is developed as a collaborative machine learning project where multiple models are trained 
        to predict airfare based on various factors like origin, destination, departure date, and cabin class.
        Each student in the project is responsible for managing a specific set of airports. Below is the breakdown:
    """)
    
    st.subheader("Student Contributions")
    
    # Display which student worked on which set of airports
    st.write("""
    **Dezhou Zhang**: Managed airports in the East Coast United States:
    - JFK: John F. Kennedy International Airport (New York City, NY)
    - EWR: Newark Liberty International Airport (Newark, NJ)
    - LGA: LaGuardia Airport (New York City, NY)
    - PHL: Philadelphia International Airport (Philadelphia, PA)
    
    **Zeyuan Wang**: Managed airports in the Southern United States:
    - ATL: Hartsfield-Jackson Atlanta International Airport (Atlanta, GA)
    - MIA: Miami International Airport (Miami, FL)
    - CLT: Charlotte Douglas International Airport (Charlotte, NC)
    - DFW: Dallas/Fort Worth International Airport (Dallas, TX)
    
    **Hao Deng**: Managed airports in the Western United States:
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

