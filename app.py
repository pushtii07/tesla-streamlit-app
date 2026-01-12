# app.py - Tesla Stock Price Prediction (Error-Free Deployment)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Tesla Stock Price Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Tesla Stock Price Forecasting System")
st.markdown("""
**An AI-powered system using LSTM deep learning models  
to forecast future Tesla stock closing prices.**
""")
st.divider()

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Forecast Settings")
horizon = st.sidebar.selectbox("Select Forecast Horizon (Days Ahead)", [1, 5, 10])

# ------------------ LOAD DATA ------------------
uploaded_file = st.file_uploader("Upload Tesla CSV file (with 'Date' and 'Close' columns)", type=["csv"])
if uploaded_file:
    df = pd.read_csv("C:\Users\HP\OneDrive\Desktop\tesla-streamlit-app\TSLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    close_prices = df["Close"].values.reshape(-1, 1)
    
    st.subheader("Recent Tesla Data")
    st.dataframe(df.tail())

    # ------------------ SCALING ------------------
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)
    last_60 = scaled_data[-60:].reshape(1, 60, 1)

    # ------------------ LOAD MODEL ------------------
    MODEL_FOLDER = "models"
    model_file = f"tesla_model_{horizon}day.h5"
    model_path = os.path.join(MODEL_FOLDER, model_file)

    if os.path.exists(model_path):
        model = load_model(model_path, compile=False)
    else:
        st.error(f"Model file not found: {model_path}")
        st.stop()

    # ------------------ PREDICTION ------------------
    scaled_prediction = model.predict(last_60)
    predicted_price = scaler.inverse_transform(scaled_prediction)[0][0]

    # ------------------ METRICS ------------------
    last_price = close_prices[-1][0]
    percent_change = ((predicted_price - last_price) / last_price) * 100

    if percent_change > 1:
        trend = "📈 Bullish"
    elif percent_change < -1:
        trend = "📉 Bearish"
    else:
        trend = "⚖️ Neutral"

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Closing Price", f"${last_price:.2f}")
    col2.metric(f"Predicted Price ({horizon} Days)", f"${predicted_price:.2f}")
    col3.metric("Market Trend", trend)

    # ------------------ CHART ------------------
    st.subheader("📊 Historical Price Trend with Forecast")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(close_prices, label="Historical Closing Price")
    ax.scatter(len(close_prices), predicted_price, color="red", s=100, label=f"{horizon}-Day Forecast")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # ------------------ INTERPRETATION ------------------
    st.subheader("🧠 Model Interpretation")
    st.markdown(f"""
    - The system predicts Tesla's **closing stock price {horizon} days ahead**.
    - Prediction is based on **past 60 trading days** using an **LSTM neural network**.
    - Current trend detected: **{trend}**.
    - This forecast helps investors understand

