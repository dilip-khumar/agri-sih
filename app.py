from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import os
import requests

# =============================
# Telegram Config (from Render environment variables)
# =============================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

app = Flask(__name__)

# =============================
# Load TFLite model
# =============================
MODEL_PATH = "tomato_model.tflite"   # make sure this file is in your repo
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =============================
# Class names
# =============================
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

# =============================
# Preprocess helper
# =============================
def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =============================
# Routes
# =============================
@app.route("/")
def home():
    return "Tomato Disease API running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    # ✅ Handle file upload from App Inventor / curl
    if not request.files:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400

    file_storage = next(iter(request.files.values()))  # take first file
    image_bytes = file_storage.read()

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # 🔽 Resize big photos to avoid 413 errors
        img.thumbnail((512, 512))
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid image: {e}"}), 400

    # Predict
    input_data = preprocess(img)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = int(np.argmax(preds))
    pred_label = class_names[pred_idx]
    pred_conf = float(preds[pred_idx] * 100)

    # ✅ Send to Telegram
    telegram_status = None
    if BOT_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        files = {"photo": ("tomato.jpg", image_bytes)}
        data = {
            "chat_id": CHAT_ID,
            "caption": f"Disease: {pred_label} ({pred_conf:.2f}%)"
        }
        try:
            r = requests.post(url, data=data, files=files, timeout=30)
            telegram_status = r.json()
        except Exception as e:
            telegram_status = str(e)

    return jsonify({
        "ok": True,
        "label": pred_label,
        "confidence": pred_conf,
        "telegram": telegram_status
    })

# =============================
# Run locally
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
