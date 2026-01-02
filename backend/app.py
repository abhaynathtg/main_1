from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

MODEL_PATH = "waste_classification_model (1).keras"
model = keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = (224, 224)  # Adjust to your model's input size

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    image_bytes = file.read()
    
    try:
        processed_image = preprocess_image(image_bytes)
        predictions = model.predict(processed_image)[0]
        
        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index])
        
        all_predictions = [
            {"class": CLASS_NAMES[i], "confidence": float(predictions[i])}
            for i in range(len(CLASS_NAMES))
        ]
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return jsonify({
            "class": predicted_class,
            "confidence": confidence,
            "all_predictions": all_predictions
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
