import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Tesla Stock Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Tesla Stock Price Prediction (5 & 10 Days)")

# ---------------- SIDEBAR ----------------
horizon = st.sidebar.selectbox(
    "Select Prediction Horizon",
    [5, 10]
)

# ---------------- LOAD DATA ----------------
uploaded_file = st.file_uploader(
    "Upload Tesla CSV file (must contain 'Close' column)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

if "Close" not in df.columns:
    st.error("CSV file must contain a 'Close' column.")
    st.stop()

close_prices = df["Close"].values.reshape(-1, 1)

# ---------------- SCALING ----------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)

if len(scaled_data) < 60:
    st.error("Dataset must have at least 60 rows.")
    st.stop()

last_60 = scaled_data[-60:].reshape(1, 60, 1)

# ---------------- MODEL LOADING ----------------
MODEL_FOLDER = "models"

MODEL_FILES = {
    5: "tesla_model_5day.h5",
    10: "tesla_model_10day.h5"
}

model_path = os.path.join(MODEL_FOLDER, MODEL_FILES[horizon])

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = load_model(model_path, compile=False)

# ---------------- PREDICTION ----------------
prediction_scaled = model.predict(last_60)
prediction = scaler.inverse_transform(prediction_scaled)[0][0]

last_price = close_prices[-1][0]
change = ((prediction - last_price) / last_price) * 100

# ---------------- OUTPUT ----------------
st.subheader(f"Prediction Result ({horizon} Days Ahead)")

st.metric(
    label="Latest Closing Price",
    value=f"${last_price:.2f}"
)

st.metric(
    label=f"Predicted Price after {horizon} Days",
    value=f"${prediction:.2f}",
    delta=f"{change:.2f}%"
)

# ---------------- DISCLAIMER ----------------
st.warning(
    "This project is for academic purposes only. "
    "Do NOT use for real trading decisions."
)
