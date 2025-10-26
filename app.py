import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorRT & TF warnings

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_json

# --- Function to safely load older .h5 models ---
def safe_load_model(path):
    try:
        return load_model(path, compile=False)
    except Exception as e:
        st.warning("⚠️ Falling back to legacy loader due to incompatibility...")
        try:
            from keras.saving import legacy_h5_format
            with open(path, "rb") as f:
                return legacy_h5_format.load_model_from_hdf5(f)
        except Exception as e2:
            st.error(f"❌ Could not load model: {e2}")
            st.stop()

# --- Load the trained CNN model safely ---
cnn_model = safe_load_model("mnist_cnn_model.h5")

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
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype("uint8"))
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

        pred = cnn_model.predict(img_array)
        digit = np.argmax(pred)
        confidence = np.max(pred) * 100

        st.success(f"✅ Predicted Digit: **{digit}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.image(img, caption="Processed Input (28x28)", width=100)
    else:
        st.warning("⚠️ Please draw a digit first!")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and TensorFlow")
