import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform

# --- Safe .h5 model loading ---
MODEL_PATH = r"D:\MNIST_DigitRecognizer\mnist_cnn_model.h5"

try:
    # Use CustomObjectScope to handle legacy models
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        cnn_model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# --- Streamlit UI ---
st.title("🖌️ MNIST Digit Recognition with Draw Feature")
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

# --- Prediction ---
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert drawing to 28x28 grayscale image
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8'))
        img = img.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

        # Predict
        pred = cnn_model.predict(img_array)
        digit = np.argmax(pred)
        st.success(f"✅ Predicted Digit: {digit}")
    else:
        st.warning("Please draw a digit first!")
