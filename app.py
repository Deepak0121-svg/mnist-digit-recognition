import os
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import requests

# -----------------------------
# Model download & load
# -----------------------------
MODEL_URL = "YOUR_DIRECT_DOWNLOAD_LINK_HERE"  # replace with your hosted .h5 URL
MODEL_PATH = "mnist_cnn_model.h5"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading MNIST model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    st.success("✅ Model downloaded successfully!")

# Load model safely
try:
    cnn_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🖌️ MNIST Digit Recognition with Draw Feature")
st.write("Draw a digit (0–9) below and click Predict!")

# Drawing canvas
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=480,
    drawing_mode="freedraw",
    key="canvas"
)

# Prediction button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas to 28x28 grayscale
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8'))
        img = img.resize((28,28))
        img = ImageOps.invert(img)
        img_array = np.array(img).reshape(1,28,28,1)/255.0

        # Predict digit
        pred = cnn_model.predict(img_array)
        digit = np.argmax(pred)
        st.success(f"✅ Predicted Digit: {digit}")
    else:
        st.warning("Please draw a digit first!")
