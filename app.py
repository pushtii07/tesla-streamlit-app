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

# ---------------------- Paths ----------------------
BASE_DIR = os.path.dirname(__file__)  # Ensures relative paths work everywhere
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_FILE = os.path.join(BASE_DIR, "TSLA.csv")

# ---------------------- Load Models ----------------------
model_paths = {
    "1-day": os.path.join(MODEL_DIR, "tesla_model_1day_lstm.h5"),
    "5-day": os.path.join(MODEL_DIR, "tesla_model_5day_lstm.h5"),
    "10-day": os.path.join(MODEL_DIR, "tesla_model_10day_lstm.h5")
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
    df = pd.read_csv(DATA_FILE)
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
    
    # Flatten if multi-day
    predicted_scaled = predicted_scaled.flatten()
    predicted_prices = s_













