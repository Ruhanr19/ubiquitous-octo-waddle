import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Power Predictor", layout="wide")

# ==============================
# LOAD MODEL FROM GOOGLE DRIVE
# ==============================
@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=1QzuNaxiIpWmfV7b4_hrMjooEEBYP4-10"
    r = requests.get(url)
    with open("model.pkl", "wb") as f:
        f.write(r.content)
    return joblib.load("model.pkl")

model = load_model()

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #1f4037, #99f2c8);
}

/* Card */
.card {
    background: #1e1e2f;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 5px 25px rgba(0,0,0,0.4);
    transition: 0.3s;
}

.card:hover {
    transform: scale(1.03);
    box-shadow: 0px 10px 35px rgba(0,0,0,0.6);
}

/* Titles */
.card h3 {
    margin-bottom: 15px;
    color: #00c6ff;
}

/* Labels */
.label {
    font-size: 14px;
    margin-top: 10px;
    color: #ccc;
}

/* Inputs */
.stNumberInput > div > div > input {
    background-color: #2b2b3c;
    color: white;
    border-radius: 10px;
}

/* Button */
div.stButton > button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    padding: 12px 25px;
    border: none;
    font-size: 16px;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.05);
}

/* Result card */
.result {
    background: #1e1e2f;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# TITLE
# ==============================
st.title("⚡ Electricity Consumption Predictor")
st.markdown("### Enter Input Values")

# ==============================
# INPUT SECTION
# ==============================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>⚡ Power Inputs</h3>", unsafe_allow_html=True)

    st.markdown('<div class="label">Reactive Power (0 - 5)</div>', unsafe_allow_html=True)
    Global_reactive_power = st.number_input("", 0.0, 5.0, 0.5, key="grp")

    st.markdown('<div class="label">Appliances (0 - 50)</div>', unsafe_allow_html=True)
    Sub_metering_1 = st.number_input("", 0.0, 50.0, 0.0, key="sm1")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>🔌 Electrical</h3>", unsafe_allow_html=True)

    st.markdown('<div class="label">Voltage (200 - 260)</div>', unsafe_allow_html=True)
    Voltage = st.number_input("", 200.0, 260.0, 230.0, key="volt")

    st.markdown('<div class="label">Room Temperature (0 - 50)</div>', unsafe_allow_html=True)
    Sub_metering_2 = st.number_input("", 0.0, 50.0, 0.0, key="sm2")

    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>🌍 Environment</h3>", unsafe_allow_html=True)

    st.markdown('<div class="label">Intensity (0 - 25)</div>', unsafe_allow_html=True)
    Global_intensity = st.number_input("", 0.0, 25.0, 5.0, key="gi")

    st.markdown('<div class="label">External Temperature (0 - 50)</div>', unsafe_allow_html=True)
    Sub_metering_3 = st.number_input("", 0.0, 50.0, 0.0, key="sm3")

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# DATE & TIME
# ==============================
st.markdown("## 📅 Date & Time")

col4, col5 = st.columns(2)

with col4:
    date = st.date_input(
        "Select Date",
        min_value=pd.to_datetime("2023-01-01"),
        max_value=pd.to_datetime("2025-12-31")
    )

with col5:
    hour = st.number_input("Hour (0 - 23)", 0, 23, 12)

day = date.day
month = date.month

# ==============================
# PREDICTION
# ==============================
if st.button("Predict ⚡"):
    try:
        input_df = pd.DataFrame([[ 
            Global_reactive_power, Voltage, Global_intensity,
            Sub_metering_1, Sub_metering_2, Sub_metering_3,
            hour, day, month
        ]], columns=[
            'Global_reactive_power','Voltage','High Voltage Appliances',
            'Sub_metering_1','Sub_metering_2','Sub_metering_3',
            'hour','day','month'
        ])

        if "index" in model.feature_names_in_:
            input_df["index"] = 0

        input_df = input_df.reindex(columns=model.feature_names_in_)

        with st.spinner("Predicting... ⚡"):
            prediction = model.predict(input_df)[0]

        st.markdown(f"""
        <div class="result">
            <h2>⚡ {prediction:.4f} kW</h2>
            <p>Energy (1 hour): {prediction:.4f} kWh</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
