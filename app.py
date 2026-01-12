import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    layout="wide"
)

st.title("📈 Tesla Stock Price Prediction")
st.markdown("### Deep Learning based Forecasting using LSTM")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Forecast Settings")

horizon = st.sidebar.selectbox(
    "Select Forecast Horizon",
    options=[1, 5, 10],
    index=0
)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    return df

try:
    df = load_data()
except Exception as e:
    st.error("❌ Unable to load TSLA.csv")
    st.stop()

# --------------------------------------------------
# Use Adj Close (as per project requirement)
# --------------------------------------------------
if "Adj Close" not in df.columns:
    st.error("❌ 'Adj Close' column not found in dataset")
    st.stop()

prices = df["Adj Close"].values.reshape(-1, 1)

# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# --------------------------------------------------
# Sequence preparation
# --------------------------------------------------
WINDOW_SIZE = 60

last_sequence = scaled_prices[-WINDOW_SIZE:]
last_sequence = last_sequence.reshape(1, WINDOW_SIZE, 1)

# --------------------------------------------------
# Load correct model
# --------------------------------------------------
MODEL_PATHS = {
    1: "models/tesla_model_1day_lstm.h5",
    5: "models/tesla_model_5day_lstm.h5",
    10: "models/tesla_model_10day_lstm.h5"
}

model_path = MODEL_PATHS[horizon]

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

model = load_model(model_path, compile=False)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
scaled_prediction = model.predict(last_sequence)

# If model predicts multiple steps (5-day / 10-day),
# take the LAST day prediction
if scaled_prediction.ndim == 2 and scaled_prediction.shape[1] > 1:
    scaled_prediction_value = scaled_prediction[0, -1]
else:
    scaled_prediction_value = scaled_prediction[0][0]

prediction = scaler.inverse_transform(
    np.array([[scaled_prediction_value]])
)[0][0]

last_actual_price = prices[-1][0]

# --------------------------------------------------
# Market trend logic
# --------------------------------------------------
if prediction > last_actual_price * 1.01:
    trend = "📈 Bullish"
elif prediction < last_actual_price * 0.99:
    trend = "📉 Bearish"
else:
    trend = "➖ Sideways"

# --------------------------------------------------
# Display prediction
# --------------------------------------------------
st.subheader(f"🔮 Predicted Closing Price ({horizon}-Day Ahead)")
st.metric(
    label="Predicted Price (USD)",
    value=f"${prediction:,.2f}",
    delta=f"{prediction - last_actual_price:,.2f}"
)

st.markdown(f"### Market Trend: **{trend}**")

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.subheader("📊 Historical Prices & Prediction")

plt.figure(figsize=(12,5))
plt.plot(df.index[-120:], prices[-120:], label="Actual Price", linewidth=2)
plt.axhline(prediction, color="red", linestyle="--", linewidth=2, label="Predicted Price")
plt.title(f"LSTM Model – {horizon}-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(alpha=0.3)

st.pyplot(plt)

# --------------------------------------------------
# Project description
# --------------------------------------------------
st.markdown("---")
st.markdown("""
### 📌 Project Overview
This project predicts **Tesla stock closing prices** using  
**Long Short-Term Memory (LSTM)** deep learning models.

**Key Features**
- Time-series forecasting using historical prices
- Predictions for **1-day, 5-day, and 10-day horizons**
- Model comparison based on forecast horizon
- Interactive Streamlit dashboard

**Tech Stack**
- Python
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- Streamlit
""")

# --------------------------------------------------
# Disclaimer
# --------------------------------------------------
st.warning(
    """
⚠️ **Disclaimer**  
This application is built strictly for **academic and learning purposes**.  
Stock prices are influenced by many unpredictable factors.  
This app **must not be used for real-world trading or financial decisions**.
"""
)



