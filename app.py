import os
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("mnist_cnn_model.h5", compile=False)

# Get sample images from folder
SAMPLE_DIR = "static/sample"
sample_images = os.listdir(SAMPLE_DIR)

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_digit = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            img = Image.open(file).convert("L")
            img = ImageOps.invert(img)
            img = img.resize((28,28))
            img_array = np.array(img).reshape(1,28,28,1)/255.0
            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
    return render_template("index.html", predicted_digit=predicted_digit, sample_images=sample_images)

if __name__ == "__main__":
    app.run(debug=True)
