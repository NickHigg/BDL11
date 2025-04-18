# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load preprocessing pipeline and model
with open("bridge_preprocessing_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)
model = load_model("bridge_ann_model.h5")

# Streamlit UI
st.set_page_config(page_title="Bridge Load Capacity Predictor", layout="centered")
st.title("🌉 Bridge Load Capacity Predictor")
st.write("Estimate the maximum load a bridge can support using an Artificial Neural Network (ANN).")

# Sidebar Inputs
st.sidebar.header("🔧 Input Bridge Features")
span = st.sidebar.slider("Span Length (ft)", 100, 1000, 300)
width = st.sidebar.slider("Deck Width (ft)", 10, 100, 30)
age = st.sidebar.slider("Age (Years)", 0, 120, 50)
lanes = st.sidebar.selectbox("Number of Lanes", [1, 2, 3, 4, 5, 6])
condition = st.sidebar.slider("Condition Rating (1 = Poor, 9 = Excellent)", 1, 9, 5)
material = st.sidebar.selectbox("Bridge Material", ["Steel", "Composite", "Concrete", "Wood", "Other"])

# Format input as DataFrame
input_data = pd.DataFrame([{
    "Span_ft": span,
    "Deck_Width_ft": width,
    "Age_Years": age,
    "Num_Lanes": lanes,
    "Condition_Rating": condition,
    "Material": material
}])

# Prediction
if st.button("🧠 Predict Max Load"):
    processed_input = pipeline.transform(input_data)
    prediction = model.predict(processed_input)[0][0]
    st.success(f"Estimated Maximum Load Capacity: **{prediction:.2f} tons**")
