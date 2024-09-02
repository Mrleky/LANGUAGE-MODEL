import streamlit as st
from googletrans import Translator
import pickle
import numpy as np

# Define load_resources function
def load_resources():
    with open('LANGUAGE DETECTION MODEL', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    return model, vectorizer, label_encoder

# Load resources
model, vectorizer, label_encoder = load_resources()
translator = Translator()

def language_detector(input_text):
    input_text_v = vectorizer.transform([input_text])
    return model.predict(input_text_v)

def predict(input_text):
    predicted_language = label_encoder.inverse_transform(language_detector(input_text))[0]
    translation = translator.translate(input_text, dest="sw")
    return predicted_language, translation.text

# Streamlit app
st.title("Language Detection and Translation")

input_text = st.text_area("Enter text here:")

if st.button("Predict"):
    if input_text:
        predicted_language, translation = predict(input_text)
        st.write(f"**Predicted Language:** {predicted_language}")
        st.write(f"**Translation to Swahili:** {translation}")
    else:
        st.error("Please enter some text for prediction.")
