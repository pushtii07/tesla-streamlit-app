import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Set Streamlit page config for professional dark blue theme
st.set_page_config(page_title="Prediction Dashboard", layout="centered", page_icon="📈")

# Load models
model_path = "models"
model_1day = load_model(os.path.join(model_path, "tesla-model_1day.h5"))
model_5day = load_model(os.path.join(model_path, "tesla-model_5day.h5"))
model_10day = load_model(os.path.join(model_path, "tesla-model_10day.h5"))

# Create three tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Models Info", "Technologies & Concepts"])

# ---------------- Tab 1: Prediction ----------------
with tab1:
    st.title("Price Prediction Dashboard")
    st.markdown("""
    Enter property/stock details below to get predictions for 1-day, 5-day, and 10-day forecasts.
    """)

    # Input features
    feature1 = st.number_input("Feature 1", value=1000)
    feature2 = st.number_input("Feature 2", value=3)
    feature3 = st.number_input("Feature 3", value=1)

    user_input = np.array([[feature1, feature2, feature3]])
    scaler = MinMaxScaler()
    scaled_input = scaler.fit_transform(user_input)

    # Predictions
    pred_1day = model_1day.predict(scaled_input)[0][0]
    pred_5day = model_5day.predict(scaled_input)[0][0]
    pred_10day = model_10day.predict(scaled_input)[0][0]

    # Display predictions
    st.subheader("Predicted Values")
    st.markdown(f"**1-Day Prediction:** {pred_1day:.2f}")
    st.markdown(f"**5-Day Prediction:** {pred_5day:.2f}")
    st.markdown(f"**10-Day Prediction:** {pred_10day:.2f}")

    # Small graph below predictions
    st.subheader("Prediction Visualization")
    days = ["1-Day", "5-Day", "10-Day"]
    predictions = [pred_1day, pred_5day, pred_10day]

    fig, ax = plt.subplots(facecolor="#0E1117")  # dark background
    ax.bar(days, predictions, color="#1f77b4")
    ax.set_ylabel("Predicted Value", color="white")
    ax.set_title("Predicted Trend", color="white")
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    st.pyplot(fig)

# ---------------- Tab 2: Models Info ----------------
with tab2:
    st.title("Models Info")
    st.markdown("""
    ### Models:
    1. **1-Day Forecast Model:** Predicts next day value.
    2. **5-Day Forecast Model:** Predicts 5-day trend.
    3. **10-Day Forecast Model:** Predicts 10-day trend.

    **Input Features:** feature1, feature2, feature3  
    **Output:** Predicted price/value  
    **Training:** Supervised learning on historical data
    """)

# ---------------- Tab 3: Technologies & Concepts ----------------
with tab3:
    st.title("Technologies & Concepts Used")
    st.markdown("""
    **Libraries:** Python, Streamlit, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn  

    **Concepts:** Data preprocessing, scaling, regression/LSTM models, visualization, deployment with Streamlit
    """)









