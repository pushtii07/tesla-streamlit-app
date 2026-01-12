import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Tesla ML Predictor", page_icon="🏎️", layout="wide")

# --- SHARED DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('TSLA.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📈 Prediction Dashboard", "📂 Project Information"])

# --- PAGE 1: PREDICTION DASHBOARD ---
if page == "📈 Prediction Dashboard":
    st.title("🏎️ Tesla (TSLA) Stock Price Prediction")
    st.markdown("This dashboard uses pre-trained **LSTM Deep Learning** models to forecast future prices.")

    # Model Selection in Sidebar
    st.sidebar.subheader("Model Settings")
    model_choice = st.sidebar.selectbox(
        "Select Forecast Horizon",
        ["1 Day Forecast", "5 Day Forecast", "10 Day Forecast"]
    )

    # Model file mapping (update paths if they are in a subfolder like 'models/')
    model_map = {
        "1 Day Forecast": "lstm_1day.h5",
        "5 Day Forecast": "lstm_5day.h5",
        "10 Day Forecast": "lstm_10day.h5"
    }

    # Load Model
    @st.cache_resource
    def get_model(path):
        try:
            return load_model(path)
        except:
            return None

    model = get_model(model_map[model_choice])

    # --- VISUALIZATION ---
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Historical Price Movement")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close Price", line=dict(color='#E81010')))
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=True, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Market Stats")
        last_price = df['Close'].iloc[-1]
        st.metric("Current TSLA Price", f"${last_price:.2f}")
        st.metric("All-Time High", f"${df['High'].max():.2f}")
        st.write(f"**Dataset End Date:** {df['Date'].max().strftime('%Y-%m-%d')}")

    # --- RUN PREDICTION ---
    st.divider()
    if st.button(f"🔮 Generate {model_choice} Prediction"):
        if model is not None:
            with st.spinner("Processing sequences through LSTM..."):
                # Placeholder for your specific preprocessing logic
                st.success(f"Model '{model_map[model_choice]}' loaded successfully!")
                st.info("The model is ready. (Add your scaling & reshaping code here to see live values).")
        else:
            st.error(f"Could not find {model_map[model_choice]}. Please ensure the file is in your GitHub repo.")

# --- PAGE 2: PROJECT INFORMATION ---
elif page == "📂 Project Information":
    st.title("Project Technical Details")
    
    st.markdown("""
    ### 🧠 Machine Learning Approach
    This project utilizes **Long Short-Term Memory (LSTM)** networks, a specialized type of Recurrent Neural Network (RNN). 
    LSTMs are designed to handle sequence data and are particularly effective for time-series forecasting because they can 'remember' long-term trends.
    
    ### 📊 Dataset Overview
    The data is sourced from your `TSLA.csv` file, containing historical trading data for Tesla Inc.
    """)

    # Show Data Sample
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

    # Dataset Metrics
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Records", len(df))
    col_b.metric("Start Date", str(df['Date'].min().date()))
    col_c.metric("End Date", str(df['Date'].max().date()))

    st.markdown("""
    ---
    **Author:** [Your Name/GitHub Handle]  
    **Frameworks:** Streamlit, TensorFlow/Keras, Pandas, Plotly.
    """)
