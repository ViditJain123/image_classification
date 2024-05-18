import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model
logging.debug("Loading the model...")
model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5")
logging.debug("Model loaded successfully.")

label_file = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
with open(label_file) as f:
    labels = f.read().splitlines()
logging.debug("Labels loaded successfully.")

def prepare_image(image, target_size):
    logging.debug("Preparing image...")
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = image.astype(np.float32)  # Ensure image is float32
    image = np.expand_dims(image, axis=0)
    logging.debug("Image prepared successfully.")
    return image

@app.route('/predict', methods=['POST'])
def predict():
    logging.debug("Received request for prediction.")
    if 'file' not in request.files:
        logging.error("No file part in the request.")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No selected file in the request.")
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logging.debug(f"Saving file to {file_path}.")
        file.save(file_path)

        logging.debug(f"File saved successfully. Opening image from {file_path}.")
        image = Image.open(file_path)
        processed_image = prepare_image(image, target_size=(224, 224))
        logging.debug("Image processed. Making predictions.")
        predictions = model(processed_image)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        label = labels[predicted_class]
        confidence = np.max(predictions)
        logging.debug(f"Prediction made: {label} with confidence {confidence}.")

        return jsonify({"class": label, "confidence": float(confidence)})

if __name__ == '__main__':
    logging.debug("Starting Flask application...")
    app.run(debug=True)
    logging.debug("Flask application started.")
