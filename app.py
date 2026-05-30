from flask import Flask, render_template, request
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

import numpy as np
from PIL import Image

app = Flask(__name__)

print("Loading model...")
model = MobileNetV2(weights="imagenet")
print("Model loaded!")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        file = request.files["image"]

        image = Image.open(file)

        image = image.resize((224, 224))

        img_array = np.array(image)

        img_array = np.expand_dims(img_array, axis=0)

        img_array = preprocess_input(img_array)

        results = model.predict(img_array, verbose=0)

        label = decode_predictions(results, top=1)[0][0]

        prediction = f"{label[1]} ({label[2]*100:.1f}%)"

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)