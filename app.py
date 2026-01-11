import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Tesla Stock Prediction", layout="wide")

st.title("📈 Tesla Stock Price Prediction")
st.markdown("### Deep Learning based Stock Forecasting (RNN & LSTM)")

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Model Settings")

model_type = st.sidebar.selectbox(
    "Select Model Architecture",
    ["LSTM", "RNN"]
)

horizon = st.sidebar.selectbox(
    "Forecast Horizon (days ahead)",
    [1, 5, 10]
)

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("TSLA.csv")
df["Close"] = df["Close"].astype(float)

prices = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

# -------------------------------
# Prepare last window
# -------------------------------
window_size = 60
last_sequence = scaled_prices[-window_size:]
last_sequence = last_sequence.reshape(1, window_size, 1)

# -------------------------------
# Load model
# -------------------------------
model_path = f"models/lstm_{horizon}day.h5"
model = load_model(model_path, compile=False)

# -------------------------------
# Prediction
# -------------------------------
scaled_prediction = model.predict(last_sequence)
prediction = scaler.inverse_transform(scaled_prediction)[0][0]

last_price = prices[-1][0]

# -------------------------------
# Market trend logic
# -------------------------------
if prediction > last_price * 1.01:
    trend = "📈 Bullish"
elif prediction < last_price * 0.99:
    trend = "📉 Bearish"
else:
    trend = "➖ Sideways"

# -------------------------------
# Display results
# -------------------------------
st.subheader(
    f"Predicted Closing Price ({horizon}-day ahead): "
    f"${prediction:.2f}"
)

st.markdown(f"### Market Trend: **{trend}**")

# -------------------------------
# Plot
# -------------------------------
st.subheader("Actual vs Prediction")

plt.figure(figsize=(10, 5))
plt.plot(prices[-100:], label="Actual Price", linewidth=2)
plt.axhline(prediction, color="red", linestyle="--", label="Predicted Price")

plt.title(f"{model_type} Model – {horizon}-Day Forecast")
plt.xlabel("Time")
plt.ylabel("Price ($)")
plt.legend()

st.pyplot(plt)

# -------------------------------
# Project description
# -------------------------------
st.markdown("---")
st.markdown("""
### 📌 Project Overview
This project uses **Recurrent Neural Networks (RNN)** and **Long Short-Term Memory (LSTM)**
models to forecast Tesla stock prices.  
Users can compare model behavior and forecast horizons in real time.

**Tech Stack:** Python, Streamlit, TensorFlow, Pandas, Matplotlib
""")


# ------------------ DISCLAIMER ------------------
st.warning(
    """
    ⚠️ **Disclaimer:**  
    This project is created for **academic and learning purposes only**.  
    Stock market prices are influenced by many external factors.  
    This application should **not be used for real financial trading decisions**.
    """
)




