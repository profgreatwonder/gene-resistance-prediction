# # from flask import Flask, request, jsonify
# # import pandas as pd
# # import pickle
# # import os

# # app = Flask(__name__)

# # # Load the Random Forest model from the pickle file
# # model_path = 'RFchloroquine (1).pkl'
# # with open(model_path, 'rb') as model_file:
# #     model = pickle.load(model_file)

# # # Endpoint to handle file uploads and make predictions

# # @app.route('/')
# # def hello():
# #     return "Hello, World!"

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     if 'file' not in request.files:
# #         return jsonify({"error": "No file part in the request"}), 400
    
# #     file = request.files['file']

# #     if file.filename == '':
# #         return jsonify({"error": "No file selected"}), 400
    
# #     if file and file.filename.endswith('.csv'):
# #         try:
# #             # Read the uploaded CSV file into a DataFrame
# #             data = pd.read_csv(file)

# #             # Make predictions using the loaded model
# #             predictions = model.predict(data)

# #             # Return the predictions as JSON
# #             return jsonify({"predictions": predictions.tolist()}), 200
# #         except Exception as e:
# #             return jsonify({"error": str(e)}), 500
# #     else:
# #         return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

# # if __name__ == '_main_':
# #     app.run(port=8080, debug=True)  # Runs on port 8080 instead of the default 5000


# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier

# # Load your trained model (replace 'PIQRF.pkl' with your model file)
# model = joblib.load('PIQRF.pkl')

# # Function to preprocess the uploaded data
# def preprocess_data(uploaded_file):
#     df = pd.read_csv(uploaded_file)
#     # Here, add the necessary preprocessing steps for your data
#     # For example, one-hot encoding or any transformations
#     # Assuming that the file contains the sequence and we one-hot encode it
#     return df

# # Function to make predictions
# def make_predictions(data, selected_drug):
#     # Here you would modify the data based on the selected drug if necessary
#     predictions = model.predict(data)
#     return predictions

# # Streamlit app
# st.title("ML Model Prediction App")

# # Upload CSV file
# uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# # Select the drug
# selected_drug = st.selectbox("Select the drug for testing", 
#                              ["Chloroquine", "Dihydroartemisinin", "Lumefantrine", "Quinine", "Halofantrine", "Piperaquine"])

# # If a file is uploaded, process and predict
# if uploaded_file is not None:
#     data = preprocess_data(uploaded_file)
#     predictions = make_predictions(data, selected_drug)
#     st.write("Predictions:")
#     st.write(predictions)

#     # Optionally save predictions to a file
#     save_predictions = st.checkbox("Save predictions to CSV")
#     if save_predictions:
#         prediction_df = pd.DataFrame(predictions, columns=["Predictions"])
#         prediction_df.to_csv('predictions.csv', index=False)
#         st.write("Predictions saved as 'predictions.csv'")





import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from Bio import SeqIO
import requests
from io import StringIO

# Load your trained model (ensure it's fitted)
model = joblib.load('PIQRF.pkl')

# # Function to preprocess the uploaded data (FASTA/FASTQ)
# def preprocess_data(file_content, file_format, expected_length=4000000):
#     sequences = []
#     for record in SeqIO.parse(file_content, file_format):
#         sequences.append(str(record.seq))
    
#     def sequence_to_numeric(seq):
#         mapping = {'A': 1, 'T': 2, 'C': 3, 'G': 4}
#         return [mapping.get(nuc, 0) for nuc in seq]
    
#     # Apply the conversion to all sequences
#     numeric_sequences = [sequence_to_numeric(seq) for seq in sequences]
    
#     # Adjust each sequence to match the expected length
#     def adjust_length(seq, target_length):
#         if len(seq) > target_length:
#             return seq[:target_length]  # Trim sequence
#         else:
#             return seq + [0] * (target_length - len(seq))  # Pad sequence with zeros
    
#     adjusted_sequences = [adjust_length(seq, expected_length) for seq in numeric_sequences]
    
#     # Convert to DataFrame
#     df = pd.DataFrame(adjusted_sequences)
#     return df


# Function to preprocess the uploaded data (FASTA/FASTQ)
def preprocess_data(file_content, file_format, expected_length=4000000):
    sequences = []
    for record in SeqIO.parse(file_content, file_format):
        sequences.append(str(record.seq))
    
    def sequence_to_numeric(seq):
        mapping = {'A': 1, 'T': 2, 'C': 3, 'G': 4}
        return [mapping.get(nuc, 0) for nuc in seq]
    
    # Apply the conversion to all sequences
    numeric_sequences = [sequence_to_numeric(seq) for seq in sequences]
    
    # Adjust each sequence to match the expected length
    def adjust_length(seq, target_length):
        if len(seq) > target_length:
            return seq[:target_length]  # Trim sequence
        else:
            return seq + [0] * (target_length - len(seq))  # Pad sequence with zeros
    
    adjusted_sequences = [adjust_length(seq, expected_length) for seq in numeric_sequences]

    # Convert to NumPy array
    encoded_sequences = np.array(adjusted_sequences)

    # Ensure it's 2D
    if encoded_sequences.ndim == 1:
        encoded_sequences = encoded_sequences.reshape(1, -1)
    elif encoded_sequences.ndim > 2:
        encoded_sequences = encoded_sequences.reshape(len(encoded_sequences), -1)

    print("Encoded sequences shape:", encoded_sequences.shape)
    
    # Convert to DataFrame
    df = pd.DataFrame(encoded_sequences)
    return df



# Streamlit app
st.title("Gene Resistance Prediction App")

# Option to upload a file or enter a URL
upload_option = st.radio("Choose how to provide the genome sequence:", ("Upload file", "Enter URL"))

if upload_option == "Upload file":
    uploaded_file = st.file_uploader("Upload Genome Sequence File", type=["fa", "fasta", "fastq"])
    if uploaded_file is not None:
        file_format = "fasta" if uploaded_file.name.endswith(".fa") or uploaded_file.name.endswith(".fasta") else "fastq"
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        data = preprocess_data(stringio, file_format)

elif upload_option == "Enter URL":
    file_url = st.text_input("Enter the URL of the Genome Sequence File")
    if file_url:
        response = requests.get(file_url)
        if response.status_code == 200:
            file_format = "fasta" if file_url.endswith(".fa") or file_url.endswith(".fasta") else "fastq"
            stringio = StringIO(response.text)
            data = preprocess_data(stringio, file_format)
        else:
            st.error("Error fetching the file from the provided URL")

# Select the drug
selected_drug = st.selectbox("Select the drug for testing", 
                             ["Chloroquine", "Dihydroartemisinin", "Lumefantrine", "Quinine", "Halofantrine", "Piperaquine"])

# Function to make predictions
def make_predictions(data, selected_drug):
    # Modify or process the data based on the selected drug if necessary
    predictions = model.predict(data)
    return predictions

# If data is available, process and predict
if 'data' in locals():
    try:
        predictions = make_predictions(data, selected_drug)
        # data_load_state = st.text('Loading data...')
        st.write("Predictions:")
        st.write(predictions)
        # data_load_state.text("Done! (using st.cache_data)")

        # Optionally save predictions to a file
        save_predictions = st.checkbox("Save predictions to CSV")
        if save_predictions:
            prediction_df = pd.DataFrame(predictions, columns=["Predictions"])
            # Adjust the index to start at 1 instead of 0
            prediction_df.index = prediction_df.index + 1
    
            # Convert DataFrame to CSV format with index titled "Serial Number"
            csv = prediction_df.to_csv(index=True, index_label="Serial Number")

            # Create a download button for the CSV file
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )

            st.write("Predictions saved as 'predictions.csv'")
    except NotFittedError as e:
        st.error(f"Model is not fitted: {str(e)}")
    except ValueError as e:
        st.error(f"Error processing data: {str(e)}")


