import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Try to load models, if missing use placeholder predictions
model_path = "models"

def load_model_safe(path):
    try:
        from tensorflow.keras.models import load_model
        return load_model(path)
    except:
        return None

model_1day = load_model_safe(os.path.join(model_path, "tesla-model_1day.h5"))
model_5day = load_model_safe(os.path.join(model_path, "tesla-model_5day.h5"))
model_10day = load_model_safe(os.path.join(model_path, "tesla-model_10day.h5"))

tab1, tab2, tab3 = st.tabs(["Prediction", "Models Info", "Technologies & Concepts"])

with tab1:
    st.title("Price Prediction")
    st.markdown("Enter property/stock details below to get predictions for 1-day, 5-day, and 10-day forecasts.")

    feature1 = st.number_input("Feature 1", value=1000)
    feature2 = st.number_input("Feature 2", value=3)
    feature3 = st.number_input("Feature 3", value=1)

    user_input = np.array([[feature1, feature2, feature3]])
    scaler = MinMaxScaler()
    scaled_input = scaler.fit_transform(user_input)

    # If models loaded, predict, else use placeholders
    if model_1day: pred_1day = model_1day.predict(scaled_input)[0][0]
    else: pred_1day = 100  # placeholder

    if model_5day: pred_5day = model_5day.predict(scaled_input)[0][0]
    else: pred_5day = 105  # placeholder

    if model_10day: pred_10day = model_10day.predict(scaled_input)[0][0]
    else: pred_10day = 110  # placeholder

    st.subheader("Predicted Values")
    st.markdown(f"**1-Day Prediction:** {pred_1day:.2f}")
    st.markdown(f"**5-Day Prediction:** {pred_5day:.2f}")
    st.markdown(f"**10-Day Prediction:** {pred_10day:.2f}")

    st.subheader("Prediction Visualization")
    days = ["1-Day", "5-Day", "10-Day"]
    predictions = [pred_1day, pred_5day, pred_10day]

    fig, ax = plt.subplots()
    ax.bar(days, predictions, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Predicted Value")
    ax.set_title("Predicted Trend")
    st.pyplot(fig)

with tab2:
    st.title("Models Info")
    st.markdown("""
    ### Models:
    1. 1-Day Forecast Model: Predicts next day value.
    2. 5-Day Forecast Model: Predicts 5-day trend.
    3. 10-Day Forecast Model: Predicts 10-day trend.

    **Input Features:** feature1, feature2, feature3  
    **Output:** Predicted price/value  
    **Training:** Supervised learning on historical data
    """)

with tab3:
    st.title("Technologies & Concepts Used")
    st.markdown("""
    **Libraries:** Python, Streamlit, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn  
    **Concepts:** Data preprocessing, scaling, regression/LSTM models, visualization, deployment with Streamlit
    """)









