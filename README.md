# 🌉 Bridge Load Capacity Predictor

This web application uses an Artificial Neural Network (ANN) to predict the maximum load (in tons) that a bridge can carry. It is designed as part of the ENCE 2530 Lab 11 assignment.

## 📊 Overview

- Trained on a dataset containing bridge characteristics
- Uses a TensorFlow-based ANN model
- Includes preprocessing with scaling and one-hot encoding
- Deployed with Streamlit for user interaction

## 🔧 Features

- Inputs:
  - Span (ft)
  - Deck width (ft)
  - Age (years)
  - Number of lanes
  - Condition rating (1–9)
  - Material type
- Outputs:
  - Predicted maximum load capacity in **tons**

## 🚀 Deployment

The app is deployed using Streamlit Cloud.

**Live App**: [your-app-link.streamlit.app](https://your-app-link.streamlit.app)

## 🛠 Files

- `app.py`: Streamlit web app
- `bridge_ann_model.h5`: Trained ANN model
- `bridge_preprocessing_pipeline.pkl`: Scikit-learn pipeline
- `requirements.txt`: Required Python libraries
