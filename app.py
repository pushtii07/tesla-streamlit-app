import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Load models
model_1day = load_model("models/tesla-model_1day.h5")
model_5day = load_model("models/tesla-model_5day.h5")
model_10day = load_model("models/tesla-model_10day.h5")

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Models Info", "Technologies Used"])

with tab1:
    st.title("Price Prediction Dashboard")

    st.markdown("Select prediction horizon and enter features to see the predicted value.")

    # Horizon selection
    horizon = st.selectbox("Select Horizon", ["1-Day", "5-Day", "10-Day"])

    # Feature inputs
    feature1 = st.number_input("Feature 1", value=1000)
    feature2 = st.number_input("Feature 2", value=3)
    feature3 = st.number_input("Feature 3", value=1)

    user_input = np.array([[feature1, feature2, feature3]])
    scaler = MinMaxScaler()
    scaled_input = scaler.fit_transform(user_input)

    # Prediction based on horizon
    if horizon == "1-Day":
        predicted_value = model_1day.predict(scaled_input)[0][0]
    elif horizon == "5-Day":
        predicted_value = model_5day.predict(scaled_input)[0][0]
    else:
        predicted_value = model_10day.predict(scaled_input)[0][0]

    # Display predicted value window
    st.subheader(f"{horizon} Prediction")
    st.markdown(f"**Predicted Value:** {predicted_value:.2f}")
    st.markdown("*Disclaimer: This prediction is based on historical data and may not reflect real future prices.*")

    # Small graph
    st.subheader("Prediction Visualization")
    horizons = ["1-Day", "5-Day", "10-Day"]
    predictions = [
        model_1day.predict(scaled_input)[0][0],
        model_5day.predict(scaled_input)[0][0],
        model_10day.predict(scaled_input)[0][0]
    ]

    fig, ax = plt.subplots()
    ax.bar(horizons, predictions, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Predicted Value")
    ax.set_title("Prediction Trend")
    st.pyplot(fig)

with tab2:
    st.title("Models Info")
    st.markdown("""
    - **1-Day Forecast Model:** Predicts next day value.  
    - **5-Day Forecast Model:** Predicts 5-day trend.  
    - **10-Day Forecast Model:** Predicts 10-day trend.  

    **Input Features:** feature1, feature2, feature3  
    **Output:** Predicted price  
    **Training:** Supervised learning on historical data
    """)

with tab3:
    st.title("Technologies Used")
    st.markdown("""
    - **Libraries:** Python, Streamlit, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn  
    - **Concepts:** Data preprocessing, scaling, regression/LSTM models, visualization, deployment with Streamlit
    """)










