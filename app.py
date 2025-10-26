# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")
st.title("🖼️ MNIST Digit Recognizer")

# -------------------------------
# Load model safely
# -------------------------------
@st.cache_resource
def load_cnn_model():
    model_path = "mnist_cnn_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Upload the H5 file in the same folder!")
        return None
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

cnn_model = load_cnn_model()

if cnn_model:
    st.success("✅ Model loaded successfully!")

    # -------------------------------
    # Image upload
    # -------------------------------
    st.sidebar.header("Upload a Digit Image")
    uploaded_file = st.sidebar.file_uploader("Choose a PNG/JPG image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Open image
        img = Image.open(uploaded_file).convert("L")  # grayscale
        img = ImageOps.invert(img)                     # invert colors if needed
        img = img.resize((28, 28))                     # resize to MNIST size

        # Convert to numpy array and normalize
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

        # Predict
        prediction = cnn_model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        # Display
        st.subheader("Uploaded Image")
        st.image(img, width=150)
        st.subheader(f"Predicted Digit: {predicted_digit}")

    # -------------------------------
    # Optional: Show sample images
    # -------------------------------
    st.sidebar.header("Sample Images")
    sample_folder = "sample_images"  # folder containing some MNIST digits
    if os.path.exists(sample_folder):
        samples = os.listdir(sample_folder)[:5]
        for s in samples:
            st.sidebar.image(os.path.join(sample_folder, s), width=70)
    else:
        st.sidebar.info("No sample images found. Create a 'sample_images' folder in the app directory.")
