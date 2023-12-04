# model.py

import joblib
import numpy as np

class HeatingCoolingModel:
    def __init__(self, model_path="model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, features):
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)
        return prediction[0]

# app.py

import streamlit as st
import pandas as pd
from model import HeatingCoolingModel

def main():
    st.title("Heating and Cooling Prediction App")

    # Sidebar for user input
    st.sidebar.header("User Input")
    temperature = st.sidebar.slider("Temperature (°C)", -10.0, 40.0, 20.0)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)

    # Load the machine learning model
    model = HeatingCoolingModel()

    # Make a prediction
    features = [temperature, humidity]
    prediction = model.predict(features)

    # Display the prediction
    st.write(f"Predicted Heating/Cooling: {prediction}")

if __name__ == "__main__":
    main()
