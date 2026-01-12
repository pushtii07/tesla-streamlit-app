# app.py - Tesla Stock Prediction Deployment
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ----------------- Paths -----------------
MODEL_FOLDER = "models"  # folder containing your .h5 models
MODEL_FILES = {
    "1-day": "tesla_model_1day_lstm.h5",
    "5-day": "tesla_model_5day_lstm.h5",
    "10-day": "tesla_model_10day_lstm.h5"
}

# ----------------- Load Models -----------------
models = {}
for key, file in MODEL_FILES.items():
    model_path = os.path.join(MODEL_FOLDER, file)
    if os.path.exists(model_path):
        models[key] = load_model(model_path)
    else:
        st.error(f"Model file not found: {model_path}")

# ----------------- Streamlit Page -----------------
st.title("Tesla Stock Price Prediction 📈")
st.write("Select prediction horizon and input recent stock data to forecast Tesla stock prices.")

# Upload CSV for recent Tesla data
uploaded_file = st.file_uploader("Upload CSV (with 'Close' prices column)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Recent Tesla Data")
    st.dataframe(df.tail())

    # Feature scaling
    scaler = MinMaxScaler(feature_range=(0,1))
    close_prices = df['Close'].values.reshape(-1,1)
    scaled_prices = scaler.fit_transform(close_prices)

    # Select prediction horizon
    horizon = st.selectbox("Select prediction horizon", ["1-day", "5-day", "10-day"])
    model = models.get(horizon)

    if model:
        # Prepare last 60 days for LSTM input
        X_test = []
        last_60_days = scaled_prices[-60:]
        X_test.append(last_60_days)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Predict
        predicted_scaled = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted_scaled)

        st.subheader(f"Predicted Tesla Price for {horizon}:")
        st.write(predicted.flatten()[0])

        # Plot
        st.subheader("Price Trend")
        plt.figure(figsize=(10,5))
        plt.plot(close_prices[-60:], color='blue', label='Recent Prices')
        plt.plot(range(60,61), predicted, color='red', marker='o', label='Prediction')
        plt.title(f"Tesla Stock Price Prediction ({horizon})")
        plt.xlabel("Days")
        plt.ylabel("Price ($)")
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("Selected model is not loaded.")
else:
    st.info("Please upload your Tesla stock CSV file to continue.")







