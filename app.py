import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Tesla Stock Price Forecasting System",
    page_icon="📈",
    layout="wide"
)

# ------------------ TITLE ------------------
st.title("📈 Tesla Stock Price Forecasting System")
st.markdown(
    """
    **An AI-powered system using LSTM deep learning models  
    to forecast future Tesla stock closing prices.**
    """
)

st.divider()

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Forecast Settings")

horizon = st.sidebar.selectbox(
    "Select Forecast Horizon (Days Ahead)",
    [1, 5, 10]
)

# ------------------ LOAD DATA ------------------
df = pd.read_csv("TSLA.csv")
df["Date"] = pd.to_datetime(df["Date"])
close_prices = df["Close"].values.reshape(-1, 1)

# ------------------ SCALING ------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)

last_60 = scaled_data[-60:].reshape(1, 60, 1)

# ------------------ LOAD MODEL ------------------
model_path = f"models/tesla_model_{horizon}day.h5"
model = load_model(model_path, compile=False)

# ------------------ PREDICTION ------------------
scaled_prediction = model.predict(last_60)
predicted_price = scaler.inverse_transform(scaled_prediction)[0][0]

# ------------------ METRICS ------------------
col1, col2, col3 = st.columns(3)

last_price = close_prices[-1][0]
price_change = predicted_price - last_price
trend = "📈 Bullish" if price_change > 0 else "📉 Bearish"

col1.metric("Latest Closing Price", f"${last_price:.2f}")
col2.metric(f"Predicted Price ({horizon} Days)", f"${predicted_price:.2f}")
col3.metric("Market Trend", trend)

# ------------------ CHART ------------------
st.subheader("📊 Historical Price Trend with Forecast")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(close_prices, label="Historical Closing Price")

ax.scatter(
    len(close_prices),
    predicted_price,
    color="red",
    s=100,
    label=f"{horizon}-Day Forecast"
)

ax.set_xlabel("Trading Days")
ax.set_ylabel("Price (USD)")
ax.legend()

st.pyplot(fig)

# ------------------ INTERPRETATION ------------------
st.subheader("🧠 Model Interpretation")

st.markdown(
    f"""
    - The system predicts Tesla's **closing stock price {horizon} days ahead**.
    - Prediction is based on **past 60 trading days** using an **LSTM neural network**.
    - Current trend detected: **{trend}**.
    - This forecast helps investors understand **short-term price direction**.
    """
)

# ------------------ MODEL INFO ------------------
with st.expander("📌 Model & Technology Details"):
    st.markdown(
        """
        **Model Used:** Long Short-Term Memory (LSTM)  
        **Why LSTM?** Effective for time-series and sequential data  
        **Input Window:** Last 60 trading days  
        **Output:** Single future closing price  
        **Frameworks:** TensorFlow, Keras, Streamlit  
        """
    )

# ------------------ DISCLAIMER ------------------
st.warning(
    """
    ⚠️ **Disclaimer:**  
    This project is created for **academic and learning purposes only**.  
    Stock market prices are influenced by many external factors.  
    This application should **not be used for real financial trading decisions**.
    """
)
