import streamlit as st
import pickle
import numpy as np
import os

# Load model (safe path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')

model = pickle.load(open(model_path, 'rb'))

st.title("🚗 Car Price Predictor")

year = st.number_input("Year", 2000, 2024)
present_price = st.number_input("Present Price(LAKHS)")
kms = st.number_input("Kms Driven")

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", [0, 1, 2, 3])

# Encoding
fuel = 1 if fuel == "Petrol" else 0
seller = 1 if seller == "Individual" else 0
transmission = 1 if transmission == "Manual" else 0

car_age = 2024 - year

if st.button("Predict"):
    input_data = np.array([[present_price, kms, owner, fuel, seller, transmission, car_age]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹ {prediction[0]:.2f} Lakhs")