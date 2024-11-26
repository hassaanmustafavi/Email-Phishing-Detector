import streamlit as st
import nltk
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model  # Import Keras load_model function

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the Email Phishing model (Logistic Regression)
with open('Email_Phishing_LG.pkl', 'rb') as f:
    tfidf_email, model_LG = pickle.load(f)

# Load the URL Phishing model
with open('URL_LR.pkl', 'rb') as f:
    tfidf_url, model_URL = pickle.load(f)

# Load the LSTM model (ensure it's saved in .h5 format)
model_LSTM = load_model('email_lstm_model.h5')

# Load stopwords for preprocessing
from nltk.corpus import stopwords
stop = stopwords.words('english')

# Function for Email preprocessing (for Email_Phishing_LG model)
def preprocess_email(email_text):
    email_text = re.sub(r'[^a-zA-Z\s]', '', email_text)  # Remove non-alphabet characters
    email_text = ' '.join([word for word in email_text.split() if word not in stop])  # Remove stopwords
    email_text = email_text.lower()  # Convert to lowercase
    email_text = re.sub(r'\d+', '', email_text)  # Remove numbers
    email_text = re.sub(r'\s+', ' ', email_text)  # Remove extra spaces
    email_text = re.sub(r'[^\w\s]', '', email_text)  # Remove special characters
    email_text = re.sub(r'http\S+', '', email_text)  # Remove URLs
    return email_text

# Function to extract and preprocess URL from the text
def extract_and_preprocess_url(text):
    # Find all URLs in the input text using a more flexible regex pattern
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    
    if not urls:
        # Attempt to find URLs with spaces or special characters
        # Fix the spaces in the URL and try matching again
        text = text.replace(" ", "")  # Remove all spaces for URL detection
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

    if urls:
        url = urls[0]  # Take the first URL (if multiple are found)
    else:
        return None  # Return None if no URL is found

    # Preprocess the URL based on training logic
    url = url.lower()  # Convert URL to lowercase
    url = re.sub(r'^https?://', '', url)  # Remove HTTP/HTTPS protocols
    domain = re.search(r'^([^/]+)', url)  # Extract domain name
    if domain:
        domain = domain.group(1)
    else:
        domain = url  # If no domain is found, use the whole URL
    
    # Tokenize URL (split by non-alphanumeric characters)
    tokens = re.split(r'\W+', domain)
    tokens_str = ' '.join(tokens)
    return tokens_str


# Streamlit GUI
st.title("Phishing Detector")

# Input for both Email Text and URL in one field
input_text = st.text_area("Enter Email Text or URL:")

if st.button("Classify as Phishing or Safe"):
    if input_text:  # If there is any input
        # Extract and preprocess the URL from the input (if exists)
        url_input = extract_and_preprocess_url(input_text)

        if url_input:  # If a URL is found
            # Vectorize the URL tokens using the same vectorizer used during training
            X_url = tfidf_url.transform([url_input])

            # Predict using the URL Phishing model
            prediction_URL = model_URL.predict(X_url)

            # Display results from the URL model only (No LSTM for URLs)
            if prediction_URL == 1:
                overall_result = "Safe"
            else:
                overall_result = "Phishing"

            # Display the results for URL input
            st.subheader("Overall Phishing Prediction Result")
            st.write(f"URL Model (Logistic Regression): {'Safe' if prediction_URL == 1 else 'Phishing'}")
            st.markdown(f"### **Result: {overall_result}**")

        else:  # If no URL found, treat the input as email text
            # Preprocess the email text
            processed_email = preprocess_email(input_text)

            # Vectorize the email text using the same vectorizer used during training
            X_email = tfidf_email.transform([processed_email])

            # Predict using the Email Phishing model
            prediction_LG = model_LG.predict(X_email)

            # Predict using the LSTM model (only for email data)
            X_email_lstm = tfidf_email.transform([processed_email])  # Use the same vectorizer
            prediction_LSTM = model_LSTM.predict(X_email_lstm)

            # Determine the overall result based on email models
            if prediction_LG == 1 and prediction_LSTM == 1:
                overall_result = "Safe"
            else:
                overall_result = "Phishing"

            # Display the results for email input
            st.subheader("Email Phishing Prediction Result")
            st.write(f"Email Model (Logistic Regression): {'Safe' if prediction_LG == 1 else 'Phishing'}")
            st.write(f"Email Model (LSTM): {'Safe' if prediction_LSTM == 1 else 'Phishing'}")
            st.markdown(f"### **Result: {overall_result}**")

    else:
        st.write("Please enter some text or a URL for prediction.")
