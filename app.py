import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import keras.src.backend.common.name_scope as ns

# --- Patch for older .h5 models ---
if not hasattr(ns, "name_scope_stack"):
    ns.name_scope_stack = []

# --- Safe model loading ---
try:
    cnn_model = tf.keras.models.load_model("mnist_cnn_model.h5", compile=False)
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

st.title("🖌️ MNIST Digit Recognition with Draw Feature")
st.write("Draw a digit (0–9) below and click Predict!")

# --- Drawing canvas ---
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
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8'))
        img = img.resize((28,28))
        img = ImageOps.invert(img)
        img_array = np.array(img).reshape(1,28,28,1)/255.0

        pred = cnn_model.predict(img_array)
        digit = np.argmax(pred)
        st.success(f"✅ Predicted Digit: {digit}")
    else:
        st.warning("Please draw a digit first!")
