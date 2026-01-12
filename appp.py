# appp.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# ----------------- Page Setup -----------------
st.set_page_config(
    page_title="Tesla Stock Prediction",
    layout="centered",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ----------------- Sidebar for Pages -----------------
page = st.sidebar.selectbox("Navigation", ["Prediction", "Technologies Used"])

# ----------------- Load Models -----------------
def load_lstm_model(model_file):
    path = os.path.join("models", model_file)
    if os.path.exists(path):
        return load_model(path, compile=False)
    else:
        st.error(f"Model {model_file} not found!")
        return None

# ----------------- Hardcoded Tesla Close Prices -----------------
tesla_close_prices = np.array([
130.5, 131.2, 132.0, 131.8, 132.5, 133.0, 134.2, 135.0,
134.8, 135.5, 136.0, 135.8, 136.5, 137.0, 138.2, 138.5,
138.0, 137.5, 138.0, 138.8, 139.0, 139.5, 140.0, 139.8,
140.5, 141.0, 141.5, 142.0, 142.5, 143.0, 143.5, 144.0,
144.5, 145.0, 145.2, 145.5, 146.0, 146.5, 147.0, 147.5,
148.0, 148.5, 149.0, 149.5, 150.0, 150.5, 151.0, 151.5,
152.0, 152.5, 153.0, 153.5, 154.0, 154.5, 155.0, 155.5,
156.0, 156.5, 157.0, 157.5
])

# Ensure minimum 60 values
if len(tesla_close_prices) < 60:
    pad_length = 60 - len(tesla_close_prices)
    tesla_close_prices = np.concatenate(
        (np.full(pad_length, tesla_close_prices[0]), tesla_close_prices)
    )

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(tesla_close_prices.reshape(-1,1))

# Prediction function
def predict_future(model, data_scaled, scaler):
    seq_length = 60
    x_input = np.array(data_scaled[-seq_length:]).reshape(1, seq_length, 1)
    yhat = model.predict(x_input, verbose=0)
    predicted = scaler.inverse_transform(yhat.reshape(-1,1))
    return predicted.flatten()[0]

# ----------------- Prediction Page -----------------
if page == "Prediction":
    st.header("Tesla Stock Price Prediction")
    
    model_1day = load_lstm_model("lstm_1day.h5")
    model_5day = load_lstm_model("lstm_5day.h5")
    model_10day = load_lstm_model("lstm_10day.h5")
    
    st.markdown("Select the prediction horizon and click Predict:")
    horizon = st.selectbox("Prediction Horizon (days)", [1,5,10])
    
    if st.button("Predict"):
        if horizon == 1:
            pred = predict_future(model_1day, data_scaled, scaler)
        elif horizon == 5:
            pred = predict_future(model_5day, data_scaled, scaler)
        else:
            pred = predict_future(model_10day, data_scaled, scaler)
        
        st.success(f"🔹 Predicted Tesla Close Price for {horizon}-Day(s): ${pred:.2f}")
        
        # ----------------- Plot -----------------
        plt.figure(figsize=(10,4))
        plt.plot(range(len(tesla_close_prices)), tesla_close_prices, label="Historical Prices", color="cyan")
        plt.scatter(len(tesla_close_prices)+horizon-1, pred, color="red", label=f"Predicted {horizon}-Day Price", s=100)
        plt.xlabel("Days", color="white")
        plt.ylabel("Price", color="white")
        plt.title("Tesla Prices and Prediction", color="white")
        plt.legend(facecolor="#222222", edgecolor="white")
        plt.grid(True, alpha=0.3, color="gray")
        plt.gca().set_facecolor("#000000")
        st.pyplot(plt)

# ----------------- Technologies Page -----------------
else:
    st.header("Technologies Used")
    
    st.markdown("""
    **1. Python:** Programming language used for the project.

    **2. Streamlit:** Used to create the interactive web app.

    **3. NumPy:** For numerical operations.

    **4. Matplotlib:** For plotting historical and predicted prices.

    **5. TensorFlow & Keras:** For building and loading LSTM models.

    **6. scikit-learn (MinMaxScaler):** To scale the data for LSTM.

    **7. LSTM Models:** Pre-trained Long Short-Term Memory neural networks for predicting Tesla stock prices.
    
    ---
    """)
    st.markdown("Use the sidebar to navigate back to the Prediction page.")

