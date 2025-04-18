import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load preprocessing pipeline
with open("bridge_preprocessing_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Load trained model (compile after loading)
model = load_model("bridge_ann_model.h5", compile=False)
model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["mae"])

st.title("🌉 Bridge Load Capacity Predictor")
st.markdown("Enter bridge features to predict **maximum load capacity (tons)** using a trained neural network model.")

# Sidebar inputs
span = st.sidebar.slider("Span (ft)", 100, 1000, 300)
deck_width = st.sidebar.slider("Deck Width (ft)", 10, 100, 30)
age = st.sidebar.slider("Bridge Age (years)", 0, 120, 50)
lanes = st.sidebar.selectbox("Number of Lanes", [1, 2, 3, 4, 5, 6])
condition = st.sidebar.slider("Condition Rating (1–9)", 1, 9, 5)
material = st.sidebar.selectbox("Material", ["Steel", "Composite", "Concrete", "Wood", "Other"])

# Build input DataFrame
user_input = pd.DataFrame([{
    "Span_ft": span,
    "Deck_Width_ft": deck_width,
    "Age_Years": age,
    "Num_Lanes": lanes,
    "Condition_Rating": condition,
    "Material": material
}])

# Predict
if st.button("Predict Load Capacity"):
    transformed_input = pipeline.transform(user_input)
    prediction = model.predict(transformed_input)[0][0]
    st.success(f"🔢 Predicted Maximum Load Capacity: **{prediction:.2f} tons**")
