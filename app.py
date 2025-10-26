import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import tensorflow as tf

# -------------------------
# Load Model safely
# -------------------------
@st.cache_resource
def load_cnn_model():
    # Load old .h5 model safely
    model = tf.keras.models.load_model("mnist_cnn_model.h5", compile=False)
    return model

cnn_model = load_cnn_model()

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🖌️")
st.title("🖌️ MNIST Digit Recognizer")
st.write("Upload an image of a digit (0-9) and let the model predict it!")

# -------------------------
# Preprocess function
# -------------------------
def preprocess_image(img):
    img = img.convert("L")                 # Grayscale
    img = ImageOps.invert(img)             # Invert colors for MNIST
    img = img.resize((28, 28))             # Resize to 28x28
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    return img_array

# -------------------------
# Upload Image
# -------------------------
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img.resize((140,140)), caption="Uploaded Image", width=140)
    
    img_array = preprocess_image(img)
    
    pred = cnn_model.predict(img_array)
    digit = np.argmax(pred)
    
    st.success(f"Predicted Digit: **{digit}**")
