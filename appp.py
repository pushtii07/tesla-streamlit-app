import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title="Tesla Stock Prediction", layout="centered")
st.title("📈 Tesla Stock Price Prediction")

# ----------------- Load Models -----------------
def load_lstm_model(model_name):
    model_path = os.path.join("models", model_name)
    if os.path.exists(model_path):
        return load_model(model_path, compile=False)
    else:
        st.error(f"Model {model_name} not found!")
        return None

model_1day = load_lstm_model("lstm_1day.h5")
model_5day = load_lstm_model("lstm_5day.h5")
model_10day = load_lstm_model("lstm_10day.h5")

# ----------------- Hardcoded 60 Tesla Close Prices -----------------
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

# ----------------- Scale the data -----------------
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(tesla_close_prices.reshape(-1,1))

# ----------------- Prediction Function -----------------
def predict_future(model, data_scaled, scaler, days):
    seq_length = 60
    temp_input = list(data_scaled[-seq_length:])
    lst_output = []

    for i in range(days):
        x_input = np.array(temp_input[-seq_length:]).reshape(1, seq_length, 1)
        yhat = model.predict(x_input, verbose=0)
        lst_output.append(yhat[0][0])
        temp_input.append(yhat[0][0])

    predicted = scaler.inverse_transform(np.array(lst_output).reshape(-1,1))
    return predicted.flatten()[-1]

# ----------------- Streamlit UI -----------------
horizon = st.selectbox("Select Prediction Horizon (days)", [1, 5, 10])

if st.button("Predict"):
    if horizon == 1:
        pred = predict_future(model_1day, data_scaled, scaler, days=1)
    elif horizon == 5:
        pred = predict_future(model_5day, data_scaled, scaler, days=5)
    else:
        pred = predict_future(model_10day, data_scaled, scaler, days=10)
    
    st.success(f"🔹 Predicted Tesla Close Price for {horizon}-Day(s): ${pred:.2f}")


