# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")

st.title("🖼️ MNIST Digit Recognizer")
st.write("Upload a handwritten digit image, and the model will predict it!")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    model_path = "mnist_cnn_model_saved"  # path to SavedModel
    model = tf.keras.models.load_model(model_path)
    return model

cnn_model = load_model()

# -------------------------
# Sidebar sample images
# -------------------------
st.sidebar.header("Sample Digits")
sample_dir = "./sample"  # folder containing sample images
sample_files = [f for f in os.listdir(sample_dir) if f.endswith(".png")]

for f in sample_files:
    img = Image.open(os.path.join(sample_dir, f))
    st.sidebar.image(img, caption=f, use_column_width=True)

# -------------------------
# Image upload
# -------------------------
uploaded_file = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")  # grayscale
    img = ImageOps.invert(img)  # invert if white digit on black background
    img = img.resize((28,28))
    img_array = np.array(img).reshape(1,28,28,1)/255.0

    # Prediction
    prediction = cnn_model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)*100

    # Display results
    st.image(img, caption="Uploaded Image", width=150)
    st.success(f"Predicted Digit: {predicted_digit}")
    st.info(f"Confidence: {confidence:.2f}%")
