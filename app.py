import os
import requests
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load environment variables from Render
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# Path to your TFLite model (make sure you included it in repo)
MODEL_PATH = "tomato_best.tflite"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define your class names (same as training dataset order)
class_names = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_YellowLeaf__Curl_Virus",
    "Tomato_mosaic_virus",
    "Tomato_healthy"
]


def preprocess_image(image_file):
    """Preprocess uploaded image for TFLite model"""
    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))  # match training size
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, image


@app.route("/")
def home():
    return "Tomato API running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image uploaded"}), 400

    file = request.files["image"]
    img_array, pil_img = preprocess_image(file)

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])

    pred_index = int(np.argmax(preds))
    label = class_names[pred_index]
    confidence = float(np.max(preds) * 100)

    # Save image temporarily to send to Telegram
    temp_path = "uploaded.jpg"
    pil_img.save(temp_path)

    telegram_status = None
    if BOT_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        files = {"photo": open(temp_path, "rb")}
        data = {"chat_id": CHAT_ID, "caption": f"Disease: {label} ({confidence:.2f}%)"}
        try:
            requests.post(url, files=files, data=data, timeout=30)
            telegram_status = "sent"
        except Exception as e:
            telegram_status = f"error: {e}"

    return jsonify({
        "ok": True,
        "label": label,
        "confidence": confidence,
        "telegram": telegram_status
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
