import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import os

# -----------------------------
# Model path (SavedModel folder)
# -----------------------------
MODEL_PATH = "mnist_cnn_model_saved"

# -----------------------------
# Load the model safely
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model folder not found at {MODEL_PATH}. Make sure you have the SavedModel folder.")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

cnn_model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🖌️")
st.title("🖌️ MNIST Digit Recognition with Draw Feature")
st.write("Draw a digit (0–9) below and click Predict!")

# -----------------------------
# Drawing canvas
# -----------------------------
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=480,
    drawing_mode="freedraw",
    key="canvas"
)

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas to 28x28 grayscale image
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8'))
        img = img.resize((28, 28))
        img = ImageOps.invert(img)  # ensure black background
        img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

        # Make prediction
        pred = cnn_model.predict(img_array)
        digit = np.argmax(pred)
        st.success(f"✅ Predicted Digit: {digit}")
    else:
        st.warning("⚠️ Please draw a digit first!")
