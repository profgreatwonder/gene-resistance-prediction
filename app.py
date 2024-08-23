# from flask import Flask, request, jsonify
# import pandas as pd
# import pickle
# import os

# app = Flask(__name__)

# # Load the Random Forest model from the pickle file
# model_path = 'RFchloroquine (1).pkl'
# with open(model_path, 'rb') as model_file:
#     model = pickle.load(model_file)

# # Endpoint to handle file uploads and make predictions

# @app.route('/')
# def hello():
#     return "Hello, World!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part in the request"}), 400
    
#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400
    
#     if file and file.filename.endswith('.csv'):
#         try:
#             # Read the uploaded CSV file into a DataFrame
#             data = pd.read_csv(file)

#             # Make predictions using the loaded model
#             predictions = model.predict(data)

#             # Return the predictions as JSON
#             return jsonify({"predictions": predictions.tolist()}), 200
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500
#     else:
#         return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

# if __name__ == '_main_':
#     app.run(port=8080, debug=True)  # Runs on port 8080 instead of the default 5000


import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load your trained model (replace 'PIQRF.pkl' with your model file)
model = joblib.load('PIQRF.pkl')

# Function to preprocess the uploaded data
def preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # Here, add the necessary preprocessing steps for your data
    # For example, one-hot encoding or any transformations
    # Assuming that the file contains the sequence and we one-hot encode it
    return df

# Function to make predictions
def make_predictions(data, selected_drug):
    # Here you would modify the data based on the selected drug if necessary
    predictions = model.predict(data)
    return predictions

# Streamlit app
st.title("ML Model Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Select the drug
selected_drug = st.selectbox("Select the drug for testing", 
                             ["Chloroquine", "Dihydroartemisinin", "Lumefantrine", "Quinine", "Halofantrine", "Piperaquine"])

# If a file is uploaded, process and predict
if uploaded_file is not None:
    data = preprocess_data(uploaded_file)
    predictions = make_predictions(data, selected_drug)
    st.write("Predictions:")
    st.write(predictions)

    # Optionally save predictions to a file
    save_predictions = st.checkbox("Save predictions to CSV")
    if save_predictions:
        prediction_df = pd.DataFrame(predictions, columns=["Predictions"])
        prediction_df.to_csv('predictions.csv', index=False)
        st.write("Predictions saved as 'predictions.csv'")
