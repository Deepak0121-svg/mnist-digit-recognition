import os
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("mnist_cnn_model.h5", compile=False)

def predict_digit(img):
    """
    Predict digit, get confidence scores and top-3 predictions.
    """
    img = img.convert("L")  # Grayscale
    img = ImageOps.invert(img)  # Invert colors if needed
    img = img.resize((28,28))
    img_array = np.array(img).reshape(1,28,28,1)/255.0

    preds = model.predict(img_array)[0]
    predicted_digit = np.argmax(preds)
    top3_indices = preds.argsort()[-3:][::-1]
    top3_scores = preds[top3_indices]

    return predicted_digit, preds, top3_indices, top3_scores

def plot_predictions(preds):
    """
    Returns a base64 PNG image of the predictions bar chart.
    """
    plt.figure(figsize=(6,4))
    plt.bar(range(10), preds, color='skyblue')
    plt.xticks(range(10))
    plt.xlabel("Digit")
    plt.ylabel("Confidence")
    plt.title("Prediction Confidence")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
    plt.close()
    return img_base64

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_digit = None
    top3 = None
    chart = None
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            img = Image.open(file)
            predicted_digit, preds, top3_indices, top3_scores = predict_digit(img)
            top3 = list(zip(top3_indices, top3_scores))
            chart = plot_predictions(preds)

    return render_template(
        "index.html",
        predicted_digit=predicted_digit,
        top3=top3,
        chart=chart
    )

if __name__ == "__main__":
    app.run(debug=True)
