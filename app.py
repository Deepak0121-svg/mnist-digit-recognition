import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import os
import requests

# --- Config ---
MODEL_PATH = "mnist_cnn_model.h5"
MODEL_URL = "YOUR_H5_FILE_DOWNLOAD_LINK"  # <-- Put your hosted .h5 URL here

# --- Download model if missing ---
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model, please wait...")
    response = requests.get(MODEL_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Model downloaded successfully!")
    else:
        st.error("Failed to download the model. Check MODEL_URL.")
        st.stop()

# --- Load model safely ---
try:
    cnn_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# --- Streamlit UI ---
st.title("🖌️ MNIST Digit Recognition")
st.write("Draw a digit (0–9) below and click Predict!")

canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=480,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert drawing to 28x28 grayscale
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8'))
        img = img.resize((28,28))
        img = ImageOps.invert(img)
        img_array = np.array(img).reshape(1,28,28,1)/255.0

        # Predict
        pred = cnn_model.predict(img_array)
        digit = np.argmax(pred)
        st.success(f"✅ Predicted Digit: {digit}")
    else:
        st.warning("Please draw a digit first!")
