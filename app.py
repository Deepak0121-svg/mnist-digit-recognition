import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorRT warnings

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer


# --- Patch for legacy InputLayer configs (ignore batch_shape etc.) ---
class FixedInputLayer(InputLayer):
    def __init__(self, **kwargs):
        # Remove deprecated args safely
        kwargs.pop("batch_shape", None)
        kwargs.pop("batch_input_shape", None)
        super().__init__(**kwargs)


# --- Safe loader with fallback ---
def safe_load_model(model_path):
    try:
        # Try normally first
        return load_model(model_path, compile=False)
    except Exception as e:
        st.warning("⚠️ Compatibility issue detected, retrying with custom InputLayer...")
        try:
            from tensorflow.keras.utils import get_custom_objects
            get_custom_objects()["InputLayer"] = FixedInputLayer
            return load_model(model_path, compile=False)
        except Exception as e2:
            st.error(f"❌ Model load failed: {e2}")
            st.stop()


# --- Load your model ---
cnn_model = safe_load_model("mnist_cnn_model.h5")

# --- Streamlit UI ---
st.set_page_config(page_title="MNIST Digit Recognition", page_icon="✏️")
st.title("🖌️ MNIST Digit Recognition with Draw Feature")
st.write("Draw a digit (0–9) below and click Predict!")

# --- Drawing canvas ---
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# --- Prediction ---
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
st.caption("Developed with ❤️ using Streamlit + TensorFlow")
