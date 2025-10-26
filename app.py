import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorRT & TF warnings

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Load the trained CNN model safely ---
try:
    cnn_model = load_model("mnist_cnn_model.h5", compile=False)
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# --- Streamlit UI setup ---
st.set_page_config(page_title="MNIST Digit Recognition", page_icon="✏️", layout="centered")

st.title("🖌️ MNIST Digit Recognition with Draw Feature")
st.write("Draw a digit (0–9) below and click **Predict!**")

# --- Drawing canvas setup ---
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=480,
    drawing_mode="freedraw",
    key="canvas",
)

# --- Prediction logic ---
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas image to 28x28 grayscale for MNIST
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype("uint8"))
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

        # Model prediction
        pred = cnn_model.predict(img_array)
        digit = np.argmax(pred)
        confidence = np.max(pred) * 100

        st.success(f"✅ Predicted Digit: **{digit}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        # Optional: Display processed image
        st.image(img, caption="Processed Input (28x28)", width=100)
    else:
        st.warning("⚠️ Please draw a digit first!")

# --- Footer ---
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and TensorFlow")
