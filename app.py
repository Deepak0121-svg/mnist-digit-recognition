# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🖼️ MNIST Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9) and the model will predict it.")

# -------------------------------
# Load the CNN model safely
# -------------------------------
MODEL_PATH = "mnist_cnn_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at {MODEL_PATH}!")
    st.stop()

try:
    cnn_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -------------------------------
# Upload image
# -------------------------------
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Open image, convert to grayscale
    img = Image.open(uploaded_file).convert("L")
    
    # Invert colors if background is black
    img = ImageOps.invert(img)
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to numpy array and normalize
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    
    # Predict
    prediction = cnn_model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    
    # Display uploaded image
    st.image(img, caption="Uploaded Image", width=150)
    
    # Display prediction
    st.subheader(f"Predicted Digit: {predicted_digit}")

    # Optional: show probabilities for each digit
    st.write("Prediction probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"Digit {i}: {prob:.4f}")
