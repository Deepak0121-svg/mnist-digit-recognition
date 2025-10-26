# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import os

st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")

st.title("🖼️ MNIST Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9) to see the prediction.")

# Load model
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("mnist_cnn_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

cnn_model = load_cnn_model()

# Sidebar for image upload
st.sidebar.header("Upload your digit image")
uploaded_file = st.sidebar.file_uploader("Choose a PNG or JPG image", type=["png","jpg","jpeg"])

# Sidebar for sample images
st.sidebar.header("Sample MNIST images")
sample_dir = "sample_images"
if os.path.exists(sample_dir):
    sample_files = os.listdir(sample_dir)
    sample_files = [f for f in sample_files if f.endswith(('.png','.jpg','.jpeg'))]
    st.sidebar.image([os.path.join(sample_dir, f) for f in sample_files[:5]], width=80, caption=[f.split('.')[0] for f in sample_files[:5]])

# Function to preprocess uploaded image
def preprocess_image(img: Image.Image):
    img = img.convert("L")                # Convert to grayscale
    img = ImageOps.invert(img)            # Invert colors
    img = img.resize((28,28))             # Resize to 28x28
    img_array = np.array(img).reshape(1,28,28,1)/255.0
    return img_array

# Predict digit
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if cnn_model:
        input_array = preprocess_image(image)
        prediction = cnn_model.predict(input_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)*100
        
        st.success(f"Predicted Digit: {digit}")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Model not loaded properly.")

# Footer
st.markdown("---")
st.write("Developed with ❤️ using TensorFlow and Streamlit")
