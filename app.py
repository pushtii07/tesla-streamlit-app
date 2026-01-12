# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# ---------------------- App Config ----------------------
st.set_page_config(page_title="Tesla Stock Prediction", layout="wide")
st.title("📈 Tesla Stock Price Prediction")
st.markdown("Predict Tesla stock price for 1-day, 5-day, and 10-day horizons using LSTM models.")

# ---------------------- Load Models ----------------------
model_paths = {
    "1-day": "models/tesla_model_1day_lstm.h5",
    "5-day": "models/tesla_model_5day_lstm.h5",
    "10-day": "models/tesla_model_10day_lstm.h5"
}

models = {}
for horizon, path in model_paths.items():
    if os.path.exists(path):
        models[horizon] = load_model(path)
    else:
        st.error(f"Model file for {horizon} not found at {path}")

# ---------------------- Load Data ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/tesla_stock.csv")  # Replace with your CSV path
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

data = load_data()
st.subheader("Historical Tesla Stock Data")
st.dataframe(data.tail(10))

# ---------------------- Prediction Function ----------------------
def predict_stock(model, data, days=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare last 60 days for prediction
    last_60 = scaled_data[-60:]
    X_test = np.array([last_60])
    predicted_scaled = model.predict(X_test)
    
    # If multi-day, flatten prediction
    predicted_scaled = predicted_scaled.flatten()
    predicted_prices = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
    return predicted_prices

# ---------------------- User Selection ----------------------
st.sidebar.header("Prediction Settings")
horizon = st.sidebar.selectbox("Select Prediction Horizon", options=["1-day", "5-day", "10-day"])

if st.sidebar.button("Predict"):
    if horizon in models:
        predictions = predict_stock(models[horizon], data, days=int(horizon.split('-')[0]))
        st.subheader(f"{horizon} Tesla Stock Price Prediction")
        pred_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=len(predictions))
        pred_df = pd.DataFrame({"Date": pred_dates, "Predicted Close": predictions.flatten()})
        st.dataframe(pred_df)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(data['Close'], label="Historical Close Price")
        plt.plot(pred_df['Date'], pred_df['Predicted Close'], label="Predicted Close Price", color='red')
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.title(f"Tesla Stock Price Prediction ({horizon})")
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("Model not loaded properly for this horizon.")











