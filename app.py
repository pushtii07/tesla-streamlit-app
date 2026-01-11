import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title("Tesla Stock Price Prediction 📈")

# Sidebar
horizon = st.sidebar.selectbox("Forecast horizon (days ahead)", [1, 5, 10])

# Load data
data = pd.read_csv("TSLA.csv")
data['Date'] = pd.to_datetime(data['Date'])
prices = data['Close'].values.reshape(-1, 1)

# Scale + sequence
scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)
last_seq = scaled[-60:]
X_input = np.expand_dims(last_seq, axis=0)

# Load horizon-specific model
try:
    model = load_model(f"models/tesla_model_{horizon}day.h5", compile=False)
except Exception as e:
    st.error(f"Could not load model for {horizon}-day horizon: {e}")
    st.stop()

# Predict
pred = model.predict(X_input)

# If horizon > 1, model should output multiple values
if pred.shape[1] >= horizon:
    pred_prices = scaler.inverse_transform(pred)[0][:horizon]
else:
    # fallback: only one prediction available
    pred_prices = [scaler.inverse_transform(pred)[0, 0]]

# Show prediction text
st.subheader(f"Predicted closing price ({horizon}-day ahead): ${pred_prices[-1]:,.2f}")

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(data['Date'][-100:], prices[-100:], label="Actual", color="blue")

# Future dates
future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=horizon+1, freq="D")
ax.plot(future_dates[1:], pred_prices, label="Prediction", color="red")

ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
