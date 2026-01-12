import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title="Tesla Stock Prediction", layout="wide")

# ----------------- Helper Functions -----------------
def load_lstm_model(model_name):
    model_path = os.path.join("models", model_name)
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error(f"Model {model_name} not found in 'models/' folder!")
        return None

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    return scaled, scaler

def predict_future(model, data_scaled, scaler, days=1):
    seq_length = 60
    temp_input = list(data_scaled[-seq_length:])
    lst_output = []

    for i in range(days):
        x_input = np.array(temp_input[-seq_length:]).reshape(1, seq_length, 1)
        yhat = model.predict(x_input, verbose=0)
        lst_output.append(yhat[0][0])
        temp_input.append(yhat[0][0])
    
    predicted = scaler.inverse_transform(np.array(lst_output).reshape(-1,1))
    return predicted.flatten()

def plot_predictions(actual, pred_1, pred_5, pred_10):
    plt.figure(figsize=(12,6))
    plt.plot(actual, label='Actual Close Price', color='blue')
    plt.plot(range(len(actual), len(actual)+len(pred_1)), pred_1, label='1-Day Forecast', color='green')
    plt.plot(range(len(actual), len(actual)+len(pred_5)), pred_5, label='5-Day Forecast', color='orange')
    plt.plot(range(len(actual), len(actual)+len(pred_10)), pred_10, label='10-Day Forecast', color='red')
    plt.xlabel("Days")
    plt.ylabel("Tesla Close Price ($)")
    plt.legend()
    st.pyplot(plt)

# ----------------- Load Models -----------------
st.sidebar.title("LSTM Models")
model_1day = load_lstm_model("lstm_1day.h5")
model_5day = load_lstm_model("lstm_5day.h5")
model_10day = load_lstm_model("lstm_10day.h5")

# ----------------- App Layout -----------------
st.title("📈 Tesla Stock Price Prediction (LSTM)")

uploaded_file = st.file_uploader("Upload Tesla Historical Stock CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv('TSLA.csv')
    st.subheader("Raw Data")
    st.dataframe(df.tail(10))

    if 'Close' not in df.columns:
        st.error("CSV must contain 'Close' column!")
    else:
        data_scaled, scaler = preprocess_data(df)
        
        # Predictions
        if model_1day and model_5day and model_10day:
            pred_1 = predict_future(model_1day, data_scaled, scaler, days=1)
            pred_5 = predict_future(model_5day, data_scaled, scaler, days=5)
            pred_10 = predict_future(model_10day, data_scaled, scaler, days=10)

            st.subheader("Predictions")
            st.write(f"🔹 1-Day Prediction: ${pred_1[-1]:.2f}")
            st.write(f"🔹 5-Day Prediction: ${pred_5[-1]:.2f}")
            st.write(f"🔹 10-Day Prediction: ${pred_10[-1]:.2f}")

            # Trend Analysis (1-day)
            trend = "Upward 📈" if pred_1[-1] > df['Close'].iloc[-1] else "Downward 📉"
            st.markdown(f"### Market Trend (1-Day): **{trend}**")

            # Plot predictions
            plot_predictions(df['Close'].values, pred_1, pred_5, pred_10)

        else:
            st.error("One or more models could not be loaded. Check 'models/' folder.")
else:
    st.info("Please upload a CSV file with Tesla historical stock prices (must include 'Close' column).")

