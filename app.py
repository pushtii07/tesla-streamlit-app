import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Tesla Stock Prediction",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Tesla Stock Price Prediction (LSTM)")
st.markdown("""
This app predicts Tesla stock prices for 1, 5, and 10 days using pre-trained LSTM models.
""")

# ----------------- Load Data -----------------
@st.cache_data
def load_stock_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ----------------- Load Models -----------------
@st.cache_resource
def load_models():
    models = {}
    models['1day'] = load_model('models/tesla_model_1day_lstm.h5')
    models['5day'] = load_model('models/tesla_model_5day_lstm.h5')
    models['10day'] = load_model('models/tesla_model_10day_lstm.h5')
    return models

# ----------------- Preprocessing -----------------
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))
    return scaled_data, scaler

def create_input_sequence(data, seq_len=60):
    X = []
    X.append(data[-seq_len:])
    return np.array(X)

# ----------------- Prediction -----------------
def predict_future(model, data, scaler, days):
    seq_len = 60
    X_input = create_input_sequence(data, seq_len)
    preds = []
    for _ in range(days):
        pred_scaled = model.predict(X_input, verbose=0)
        preds.append(pred_scaled[0,0])
        # Slide window
        X_input = np.append(X_input[:,1:,:], [[pred_scaled[0,0]]], axis=1)
    preds_actual = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    return preds_actual.flatten()

# ----------------- Main App -----------------
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload Tesla Stock CSV", type=["csv"], help="Upload TSLA.csv with Date & Close columns")

if uploaded_file:
    df = load_stock_data(uploaded_file)
    st.subheader("📊 Historical Stock Prices")
    st.dataframe(df.tail(10))

    # Closing price chart
    st.subheader("📈 Closing Price Chart")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['Date'], df['Close'], color='blue', label='Closing Price')
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # Load models
    models = load_models()
    
    # Preprocess data
    scaled_data, scaler = preprocess_data(df)

    # Predict
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

    # Plot predicted values
    st.subheader("📊 Predicted Prices")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(range(1,6), pred_5, marker='o', label='5-Day Pred')
    ax2.plot(range(1,11), pred_10, marker='o', label='10-Day Pred')
    ax2.scatter(1, pred_1[0], color='red', label='1-Day Pred', zorder=5)
    ax2.set_xlabel("Days Ahead")
    ax2.set_ylabel("Predicted Price (USD)")
    ax2.legend()
    st.pyplot(fig2)

    # Trend Analysis
    st.subheader("📈 Market Trend Analysis")
    last_price = df['Close'].values[-1]
    trend_msg = "🔺 Uptrend" if pred_1[0] > last_price else "🔻 Downtrend"
    st.markdown(f"### Expected trend for next day: **{trend_msg}**")

else:
    st.info("Please upload your Tesla stock CSV file to continue.")










