import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Power Predictor", layout="centered")

st.title("⚡ Electricity Consumption Predictor")

# Load model
model = joblib.load("model.pkl")

st.markdown("### Enter Input Values")

# ==============================
# COLUMNS
# ==============================


# ==============================
# COLUMNS (MANUAL INPUT)
# ==============================
col1, col2, col3 = st.columns(3)

with col1:
    Global_reactive_power = st.number_input(
        "Reactive Power (0 - 5)",
        min_value=0.0, max_value=5.0, value=0.5
    )
    Sub_metering_1 = st.number_input(
        "Number of appliances (0 - 50)",
        min_value=0.0, max_value=50.0, value=0.0
    )

with col2:
    Voltage = st.number_input(
        "Voltage (200 - 260)",
        min_value=200.0, max_value=260.0, value=230.0
    )
    Sub_metering_2 = st.number_input(
        "Room Temperature (0 - 50)",
        min_value=0.0, max_value=50.0, value=0.0
    )

with col3:
    Global_intensity = st.number_input(
        "Intensity (0 - 50)",
        min_value=0.0, max_value=50.0, value=5.0
    )
    Sub_metering_3 = st.number_input(
        "External Temperature (0 - 50)",
        min_value=0.0, max_value=50.0, value=0.0
    )
# ==============================
# DATE INPUT (MANUAL + RANGE)
# ==============================
st.markdown("### 📅 Enter Date")

date = st.date_input(
    "Select Date",
    min_value=pd.to_datetime("2007-01-01"),
    max_value=pd.to_datetime("2010-12-31")
)

# Extract features
hour = st.slider("Hour", 0, 23, 12)
day = date.day
month = date.month

# ==============================
# PREDICTION
# ==============================

# ==============================
# PREDICTION
# ==============================

if st.button("Predict"):
    try:
        st.write("Button clicked!")  # debug

        input_df = pd.DataFrame([[ 
            Global_reactive_power, Voltage, Global_intensity,
            Sub_metering_1, Sub_metering_2, Sub_metering_3,
            hour, day, month
        ]], columns=[
            'Global_reactive_power','Voltage','Global_intensity',
            'Sub_metering_1','Sub_metering_2','Sub_metering_3',
            'hour','day','month'
        ])

        # Fix missing columns
        if "index" in model.feature_names_in_:
            input_df["index"] = 0

        input_df = input_df.reindex(columns=model.feature_names_in_)

        st.write("Input DF:", input_df)  # debug

        prediction = model.predict(input_df)[0]

        st.success(f"⚡ Power Consumption: {prediction:.4f} kW")
        st.info(f"🔋 Energy (1 hour): {prediction:.4f} kWh")

    except Exception as e:
        st.error(f"Error: {e}")