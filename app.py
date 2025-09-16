# app.py
import os
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import requests

# Try tflite_runtime first, fallback to tensorflow
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except Exception:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

MODEL_PATH = "tomato_best.tflite"   # put your model here
CLASS_NAMES = [
    "Tomato_Healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Bacterial_spot"
    # edit to EXACT order used during training
]

# Telegram (fill via Render environment variables)
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID   = os.environ.get("CHAT_ID")

# Load tflite
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(img, size=(224,224)):
    img = img.convert("RGB").resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def run_inference(pil_image):
    x = preprocess(pil_image)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds)) * 100.0
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"
    return label, conf, preds.tolist()

def send_to_telegram(image_bytes, caption):
    if not BOT_TOKEN or not CHAT_ID:
        return None
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {"photo": ("leaf.jpg", image_bytes, "image/jpeg")}
    data = {"chat_id": CHAT_ID, "caption": caption}
    resp = requests.post(url, data=data, files=files, timeout=30)
    return resp

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Tomato API running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    # accept form field 'image' (multipart/form-data)
    if 'image' not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400
    f = request.files['image']
    try:
        img = Image.open(BytesIO(f.read())).convert("RGB")
    except Exception as e:
        return jsonify({"ok": False, "error": f"Invalid image: {e}"}), 400

    label, conf, preds = run_inference(img)
    caption = f"Disease: {label} ({conf:.2f}%)"

    # send to Telegram (async not necessary for beginner)
    send_resp = send_to_telegram(f.stream.read(), caption) if (BOT_TOKEN and CHAT_ID) else None

    return jsonify({"ok": True, "label": label, "confidence": conf, "telegram": (send_resp.status_code if send_resp else None)})
