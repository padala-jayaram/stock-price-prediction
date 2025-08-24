# app.py — Streamlit version of your stock price predictor

import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
import streamlit as st

# ===============================
# Load Scaler and Model
# ===============================
@st.cache_resource
def load_scaler():
    return pickle.load(open("scaler_fitted.pkl", "rb"))

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("stock_prediction_model.keras")

scaler = load_scaler()
model = load_model()

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈")
st.title("📈 Stock Price Predictor using LSTM")

st.markdown("Enter stock details below to predict the **Close Price**:")

# Input fields for features
open_price = st.number_input("Open Price", value=0.0, step=1.0)
high_price = st.number_input("High Price", value=0.0, step=1.0)
low_price = st.number_input("Low Price", value=0.0, step=1.0)
last_price = st.number_input("Last Price", value=0.0, step=1.0)

predict_btn = st.button("🔮 Predict Close Price")

# ===============================
# Prediction Logic
# ===============================
if predict_btn:
    try:
        # Define features list
        features = ["Open", "High", "Low", "Last"]

        # Build new entry
        new_entry_data = [[open_price, high_price, low_price, last_price]]
        new_entry_df = pd.DataFrame(new_entry_data, columns=features)

        # Scale
        new_scaled = scaler.transform(new_entry_df)

        # Reshape for LSTM input (samples, timesteps, features=1)
        new_scaled = new_scaled.reshape(new_scaled.shape[0], new_scaled.shape[1], 1)

        # Predict
        predicted_scaled = model.predict(new_scaled)

        # Pad prediction to match feature count for inverse transform
        predicted_padded = np.concatenate(
            [predicted_scaled, np.zeros((predicted_scaled.shape[0], len(features) - 1))],
            axis=1,
        )

        # Inverse transform
        predicted_price = scaler.inverse_transform(predicted_padded)[:, 0][0]

        st.success(f"✅ Predicted Close Price: {predicted_price:.2f}")

    except Exception as e:
        st.error("Prediction failed. Please check scaler/model compatibility.")
        st.exception(e)
