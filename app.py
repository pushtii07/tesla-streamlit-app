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
data = pd.read_csv("tesla.csv")
data['Date'] = pd.to_datetime(data['Date'])
prices = data['Close'].values.reshape(-1, 1)

# Scale + sequence
scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)
last_seq = scaled[-60:]
X_input = np.expand_dims(last_seq, axis=0)


model = load_model(f"models/tesla_model_{horizon}day.h5")
pred = model.predict(X_input)
pred_price = scaler.inverse_transform(pred)[0, 0]

st.subheader(f"Predicted closing price ({horizon}-day ahead): ${pred_price:,.2f}")


fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(data['Date'][-100:], prices[-100:], label="Actual", color="blue")
future_dates = [data['Date'].iloc[-1 + i] for i in range(horizon + 1)]
ax.plot(future_dates, [prices[-1][0]] + [pred_price], label="Prediction", color="red")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)
