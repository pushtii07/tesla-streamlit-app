import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Tesla Stock Prediction",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Tesla Stock Price Prediction (LSTM)")
st.markdown("Predict Tesla stock prices for 1, 5, and 10 days using pre-trained LSTM models.")

# ----------------- Paths & Models -----------------
MODEL_FOLDER = "models"
MODEL_FILES = {
    "1day": "tesla_model_1day_lstm.h5",
    "5day": "tesla_model_5day_lstm.h5",
    "10day": "tesla_model_10day_lstm.h5"
}

# ----------------- Helper Functions -----------------
def check_models_exist():
    missing = []
    for key, file in MODEL_FILES.items():
        path = os.path.join(MODEL_FOLDER, file)
        if not os.path.isfile(path):
            missing.append(file)
    return missing

def load_models():
    models = {}
    for key, file in MODEL_FILES.items():
        path = os.path.join(MODEL_FOLDER, file)
        try:
            models[key] = load_model(path)
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
            st.stop()
    return models

def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))
    return scaled_data, scaler

def create_input_sequence(data, seq_len=60):
    """Create a single sequence from the last 'seq_len' data points"""
    return np.array([data[-seq_len:]])

def predict_future(model, data, scaler, days):
    seq_len = 60
    X_input = create_input_sequence(data, seq_len)
    preds = []
    for _ in range(days):
        pred_scaled = model.predict(X_input, verbose=0)
        preds.append(pred_scaled[0,0])
        # Append the new prediction and remove the first element to maintain sequence
        X_input = np.append(X_input[:,1:,:], [[pred_scaled[0,0]]], axis=1)
    preds_actual = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    return preds_actual.flatten()

# ----------------- Sidebar -----------------
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader(
    "Upload Tesla Stock CSV", 
    type=["csv"], 
    help="CSV must have 'Date' and 'Close' columns."
)

# ----------------- Check Models -----------------
missing_models = check_models_exist()
if missing_models:
    st.error(f"❌ Missing model files in '{MODEL_FOLDER}': {missing_models}")
    st.stop()

# ----------------- Load Models -----------------
models = load_models()

# ----------------- Main App -----------------
if uploaded_file:
    # Read and process data
    df = pd.read_csv(uploaded_file)
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("CSV must contain 'Date' and 'Close' columns.")
        st.stop()
        
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    st.subheader("📊 Historical Stock Prices")
    st.dataframe(df.tail(10))

    # Plot historical prices
    st.subheader("📈 Closing Price Chart")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['Date'], df['Close'], color='blue', label='Close Price')
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # Preprocess data
    scaled_data, scaler = preprocess_data(df)

    # Predictions
    st.subheader("🤖 Predictions")
    col1, col2, col3 = st.columns(3)

    with col1:
        pred_1 = predict_future(models['1day'], scaled_data, scaler, 1)
        st.metric("1-Day Prediction", f"${pred_1[0]:.2f}")

    with col2:
        pred_5 = predict_future(models['5day'], scaled_data, scaler, 5)
        st.metric("5-Day Prediction", f"${pred_5[-1]:.2f}")

    with col3:
        pred_10 = predict_future(models['10day'], scaled_data, scaler, 10)
        st.metric("10-Day Prediction", f"${pred_10[-1]:.2f}")

    # Plot predictions
    st.subheader("📊 Predicted Prices")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(range(1,6), pred_5, marker='o', label='5-Day Prediction')
    ax2.plot(range(1,11), pred_10, marker='o', label='10-Day Prediction')
    ax2.scatter(1, pred_1[0], color='red', label='1-Day Prediction', zorder=5)
    ax2.set_xlabel("Days Ahead")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    st.pyplot(fig2)

    # Trend analysis
    st.subheader("📈 Market Trend")
    last_price = df['Close'].values[-1]
    trend_msg = "🔺 Uptrend expected" if pred_1[0] > last_price else "🔻 Downtrend expected"
    st.markdown(f"### {trend_msg}")

else:
    st.info("Please upload your TSLA CSV file to continue.")









