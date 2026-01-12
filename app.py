import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Tesla Stock Prediction",
    layout="wide"
)

# ==================================================
# DARK THEME (CUSTOM CSS)
# ==================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e5e7eb;
}
[data-testid="stHeader"] {
    background-color: #020617;
}
[data-testid="stSidebar"] {
    background-color: #020617;
}
h1, h2, h3, h4 {
    color: #f9fafb !important;
}
p, span, label {
    color: #d1d5db !important;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    return df

df = load_data()

prices = df["Adj Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

WINDOW_SIZE = 60
last_sequence = scaled_prices[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)

# ==================================================
# LOAD MODELS
# ==================================================
MODEL_PATHS = {
    ("LSTM", 1): "models/tesla_model_1day_lstm.h5",
    ("LSTM", 5): "models/tesla_model_5day_lstm.h5",
    ("LSTM", 10): "models/tesla_model_10day_lstm.h5",
}

# ==================================================
# TOP NAVIGATION TABS (LIKE WEBSITE)
# ==================================================
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Project Overview", "ℹ️ About Project"])

# ==================================================
# TAB 1 — PREDICTION PAGE
# ==================================================
with tab1:
    st.title("📈 Tesla Stock Price Prediction")

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox("Select Model", ["LSTM"])
    with col2:
        horizon = st.selectbox("Forecast Horizon (Days)", [1, 5, 10])

    model_path = MODEL_PATHS[(model_type, horizon)]
    model = load_model(model_path, compile=False)

    scaled_pred = model.predict(last_sequence)

    if scaled_pred.ndim == 2 and scaled_pred.shape[1] > 1:
        scaled_pred_value = scaled_pred[0, -1]
    else:
        scaled_pred_value = scaled_pred[0][0]

    prediction = scaler.inverse_transform([[scaled_pred_value]])[0][0]
    last_price = prices[-1][0]

    if prediction > last_price * 1.01:
        trend = "📈 Bullish"
    elif prediction < last_price * 0.99:
        trend = "📉 Bearish"
    else:
        trend = "➖ Sideways"
delta_value = prediction - last_price
delta_color = "#22c55e" if delta_value >= 0 else "#ef4444"

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #020617, #111827);
    padding: 25px;
    border-radius: 14px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.6);
">
    <h3 style="color:#9ca3af; margin-bottom:10px;">
        Predicted Price ({horizon}-Day Ahead)
    </h3>
    <h1 style="color:#f9fafb; font-size:52px; margin:0;">
        ${prediction:,.2f}
    </h1>
    <p style="color:{delta_color}; font-size:18px; margin-top:8px;">
        Δ {delta_value:,.2f} USD from last close
    </p>
</div>
""", unsafe_allow_html=True)


    st.markdown(f"### Market Trend: **{trend}**")

# ==================================================
# TAB 2 — PROJECT OVERVIEW + GRAPH
# ==================================================
with tab2:
    st.title("📊 Project Overview")

    st.markdown("""
    ### Tesla Stock Price Prediction using Deep Learning

    This project focuses on predicting **Tesla stock closing prices**
    using **Recurrent Neural Networks (RNN)** and **Long Short-Term Memory (LSTM)** models.

    **Key Highlights**
    - Time-series forecasting
    - 1-day, 5-day, 10-day prediction
    - LSTM based deep learning models
    - Interactive Streamlit dashboard
    """)

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df.index[-120:], prices[-120:], label="Actual Price", linewidth=2)
        ax.set_title("Tesla Historical Closing Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

# ==================================================
# TAB 3 — ABOUT PROJECT
# ==================================================
with tab3:
    st.title("ℹ️ About This Project")

    st.markdown("""
    ### Domain
    **Financial Services / Stock Market Analysis**

    ### Technologies Used
    - Python
    - TensorFlow / Keras
    - Pandas, NumPy
    - Scikit-learn
    - Streamlit

    ### Business Use Cases
    - Stock trend analysis
    - Investment planning
    - Financial forecasting
    - Educational deep learning applications

    ### Disclaimer
    ⚠️ This project is created **only for academic and learning purposes**.
    It must **not** be used for real-world trading decisions.
    """)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown(
    "<center>🚀 Tesla Stock Prediction Project | Deep Learning & Streamlit</center>",
    unsafe_allow_html=True
)









