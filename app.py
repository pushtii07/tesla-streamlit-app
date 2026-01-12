import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ---------------- Safe model loading ----------------
def load_model_safe(path):
    try:
        from tensorflow.keras.models import load_model
        return load_model(path)
    except Exception as e:
        print(f"Could not load {path}: {e}")
        return None

# Use your actual filenames
model_1day = load_model_safe("models/tesla_model_1day_lstm.h5")
model_5day = load_model_safe("models/tesla_model_5day_lstm.h5")
model_10day = load_model_safe("models/tesla_model_10day_lstm.h5")

# ---------------- Create tabs ----------------
tab1, tab2, tab3 = st.tabs(["Prediction", "Models Info", "Technologies Used"])

# ---------------- Tab 1: Prediction Window ----------------
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

    # ---------------- Predictions with safe fallback ----------------
    if model_1day:
        pred_1day = model_1day.predict(scaled_input)[0][0]
    else:
        pred_1day = 100  # placeholder

    if model_5day:
        pred_5day = model_5day.predict(scaled_input)[0][0]
    else:
        pred_5day = 105  # placeholder

    if model_10day:
        pred_10day = model_10day.predict(scaled_input)[0][0]
    else:
        pred_10day = 110  # placeholder

    # Select prediction based on horizon
    if horizon == "1-Day":
        predicted_value = pred_1day
    elif horizon == "5-Day":
        predicted_value = pred_5day
    else:
        predicted_value = pred_10day

    # Display predicted value
    st.subheader(f"{horizon} Prediction")
    st.markdown(f"**Predicted Value:** {predicted_value:.2f}")
    st.markdown("*Disclaimer: This prediction is based on historical data and may not reflect real future prices.*")

    # Graph showing all horizons
    st.subheader("Prediction Visualization")
    horizons = ["1-Day", "5-Day", "10-Day"]
    predictions = [pred_1day, pred_5day, pred_10day]

    fig, ax = plt.subplots()
    ax.bar(horizons, predictions, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Predicted Value")
    ax.set_title("Prediction Trend")
    st.pyplot(fig)

# ---------------- Tab 2: Models Info ----------------
with tab2:
    st.title("Models Info")
    st.markdown("""
    - **1-Day Forecast Model:** Predicts next day value.  
    - **5-Day Forecast Model:** Predicts 5-day trend.  
    - **10-Day Forecast Model:** Predicts 10-day trend.  

    **Input Features:** feature1, feature2, feature3  
    **Output:** Predicted price  
    **Training:** LSTM on historical Tesla stock / property data
    """)

# ---------------- Tab 3: Technologies Used ----------------
with tab3:
    st.title("Technologies & Concepts Used")
    st.markdown("""
    - **Libraries:** Python, Streamlit, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn  
    - **Concepts:** Data preprocessing, scaling, LSTM models, visualization, deployment with Streamlit
    """)











