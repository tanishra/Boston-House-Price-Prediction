import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open("model.pkl", 'rb') as file:
    model = pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title("🏡 Boston House Price Prediction App")

# Collect user input
CRIM = st.number_input("CRIM (Crime rate)", min_value=0.0, value=0.1)
ZN = st.number_input("ZN (Zoned land proportion)", min_value=0.0, value=0.0)
INDUS = st.number_input("INDUS (Business acres)", min_value=0.0, value=5.0)
CHAS = st.selectbox("CHAS (Bounds River: 1=Yes, 0=No)", [0, 1])
NOX = st.number_input("NOX (Pollution Level)", min_value=0.0, value=0.5)
RM = st.number_input("RM (Avg. Rooms per House)", min_value=1.0, value=6.0)
AGE = st.number_input("AGE (Older Houses Proportion)", min_value=0.0, value=50.0)
DIS = st.number_input("DIS (Distance to Employment)", min_value=0.0, value=4.0)
RAD = st.number_input("RAD (Highway Accessibility)", min_value=1, value=5)
TAX = st.number_input("TAX (Property Tax)", min_value=0, value=300)
PTRATIO = st.number_input("PTRATIO (Pupil-Teacher Ratio)", min_value=1.0, value=15.0)
B = st.number_input("B (Diversity Score)", min_value=0.0, value=400.0)
LSTAT = st.number_input("LSTAT (Low Income %)", min_value=0.0, value=10.0)

# Predict button
if st.button("Predict Price 💰"):
    # Convert inputs into an array
    features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
    
    # Apply the same transformation as training
    features_scaled = scaler.transform(features)
    
    # Predict using the trained model
    prediction = model.predict(features_scaled)
    
    # Display prediction
    st.success(f"🏠 Predicted House Price: **${prediction[0]*1000:,.2f}**")


