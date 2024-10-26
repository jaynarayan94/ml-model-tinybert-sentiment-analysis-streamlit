import streamlit as st
import os
import torch
import pandas as pd
from transformers import pipeline
import boto3

# Initialize S3 client and specify bucket details
bucket_name = "mlops-jaynd"
local_path = 'tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'
s3 = boto3.client('s3')

# Function to download model from S3
def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                s3.download_file(bucket_name, s3_key, local_file)

# Initialize Streamlit app UI elements
st.title("Machine Learning Model Deployment at the Server")

# Button to download the model
button = st.button("Download Model")
if button:
    with st.spinner("Downloading... Please wait!"):
        download_dir(local_path, s3_prefix)
        st.success("Model downloaded successfully!")

# Initialize text input area for single prediction
text = st.text_area("Enter Your Review", "Type here...")
predict = st.button("Predict")

# Load the model for predictions
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
classifier = pipeline('text-classification', model=local_path, device=device)

# Single prediction
if predict:
    with st.spinner("Predicting..."):
        output = classifier(text)
        output = classifier(text)
        st.write(output)

# Bulk prediction section
st.subheader("Bulk Prediction")
uploaded_file = st.file_uploader("Upload a CSV/Excel file with a column named 'text' for predictions", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Load the file into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    
    # Check if 'text' column exists
    if 'text' not in data.columns:
        st.error("The uploaded file must contain a 'text' column.")
    else:
        with st.spinner("Running bulk predictions..."):
            # Perform predictions
            data['prediction'] = data['text'].apply(lambda x: classifier(x)[0]['label'])
            data['probability'] = data['text'].apply(lambda x: classifier(x)[0]['score'])
            
            # Show first five predictions
            st.write("Sample Predictions(Initial 5 records of the uploaded file):", data[['text', 'prediction', 'probability']].head())
            
            # Convert DataFrame to CSV for download
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime='text/csv'
            )
