import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.exceptions import NotFittedError
from Bio import SeqIO
import requests
from io import StringIO

# Load your trained models (ensure each one is fitted)
chloroquine_phenotype_model = joblib.load('./pickle_files/chloroquine_phenotype.pkl')
DHA_phenotype_model = joblib.load('./pickle_files/DHA_phenotype.pkl')
HFL_phenotype_model = joblib.load('./pickle_files/HFL_phenotype.pkl')
LUM_phenotype_model = joblib.load('./pickle_files/LUM_phenotype.pkl')
PIQ_phenotype_model = joblib.load('./pickle_files/PIQ_phenotype.pkl')
quinine_phenotype_model = joblib.load('./pickle_files/quinine_phenotype.pkl')

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

# Map the drug names to their respective models
drug_model_mapping = {
    "Chloroquine": chloroquine_phenotype_model,
    "Dihydroartemisinin": DHA_phenotype_model,
    "Lumefantrine": LUM_phenotype_model,
    "Quinine": quinine_phenotype_model,
    "Halofantrine": HFL_phenotype_model,
    "Piperaquine": PIQ_phenotype_model
}

# Function to make predictions
def make_predictions(data, selected_drug):
    # Get the corresponding model for the selected drug
    model = drug_model_mapping.get(selected_drug)
    
    if model is None:
        st.error("Model for the selected drug is not available.")
        return None

    # Make predictions with the selected model
    predictions = model.predict(data)
    return predictions

# If data is available, process and predict
if 'data' in locals():
    try:
        predictions = make_predictions(data, selected_drug)
        if predictions is not None:
            st.write("Predictions:")
            st.write(predictions)

            # Optionally save predictions to a file
            save_predictions = st.checkbox("Save predictions to CSV")
            if save_predictions:
                prediction_df = pd.DataFrame(predictions, columns=["Predictions"])
                prediction_df.index = prediction_df.index + 1  # Adjust the index to start at 1
                
                csv = prediction_df.to_csv(index=True, index_label="Serial Number")
                
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )

    except NotFittedError as e:
        st.error(f"Model is not fitted: {str(e)}")
    except ValueError as e:
        st.error(f"Error processing data: {str(e)}")
