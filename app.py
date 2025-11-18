from flask import Flask, render_template, request, jsonify
import os
import base64
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import cv2

# ======================================
# 🔥 YOLO
# ======================================
from ultralytics import YOLO

# ======================================
# 🔥 IMPORT RNN / LSTM / GRU (Keras)
# ======================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Embedding, Bidirectional
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

# =======================================================
# 🔥 Load YOLO Model (otomatis download jika belum ada)
# =======================================================
try:
    MODEL_YOLO = YOLO("yolov8n.pt")  # otomatis download
    print("✅ YOLO model loaded")
except Exception as e:
    MODEL_YOLO = None
    print("❌ YOLO model gagal load:", e)

# ======================================================================
# 🏠 HALAMAN UTAMA
# ======================================================================
@app.route('/')
def index():
    return render_template('index.html')


# ======================================================================
# 🔥 LOGIC GATE
# ======================================================================
@app.route('/logic')
def logic_page():
    return render_template('kalkulator.html')


@app.route('/api/logic', methods=['POST'])
def logic_gate():
    data = request.json or {}
    a = int(data.get('a', 0))
    b = int(data.get('b', 0))
    gate = data.get('gate', 'AND').upper()

    gates = {
        "AND": lambda x, y: x & y,
        "OR": lambda x, y: x | y,
        "XOR": lambda x, y: x ^ y,
        "NAND": lambda x, y: int(not (x & y)),
        "NOR": lambda x, y: int(not (x | y)),
        "XNOR": lambda x, y: int(not (x ^ y))
    }

    result = gates[gate](a, b)

    truth_table = [
        {'A': x, 'B': y, 'Output': gates[gate](x, y)}
        for x in [0, 1]
        for y in [0, 1]
    ]

    return jsonify({'ok': True, 'result': result, 'truth_table': truth_table})


# ======================================================================
# TEXT GENERATOR
# ======================================================================
@app.route('/predict_word')
def predict_word_page():
    return render_template('generate.html')


@app.route('/api/generate', methods=['POST'])
def generate_text_api():
    data = request.json or {}

    raw_text = data.get("text", "")
    model_type = data.get("model", "LSTM")
    seed = data.get("seed", "")
    gen_length = int(data.get("length", 100))
    epochs = int(data.get("epochs", 50))

    if len(raw_text) < 20:
        return jsonify({"ok": False, "error": "Teks terlalu pendek."})

    text = raw_text.lower()
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    seq_len = 40
    X, y = [], []

    for i in range(0, len(text) - seq_len):
        seq = text[i:i + seq_len]
        next_char = text[i + seq_len]
        X.append([char_to_idx[c] for c in seq])
        y.append(char_to_idx[next_char])

    X = np.array(X)
    y = to_categorical(y, num_classes=len(chars))

    model = Sequential()
    model.add(Embedding(len(chars), 64, input_length=seq_len))

    if model_type == "LSTM":
        model.add(LSTM(128))
    elif model_type == "GRU":
        model.add(GRU(128))
    elif model_type == "Bidirectional RNN":
        model.add(Bidirectional(LSTM(128)))
    else:
        model.add(SimpleRNN(128))

    model.add(Dense(len(chars), activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    history = model.fit(X, y, epochs=epochs, batch_size=128, verbose=0)

    seed = seed.lower().rjust(seq_len)[:seq_len]
    generated = seed

    for _ in range(gen_length):
        x_pred = np.zeros((1, seq_len))
        for i, c in enumerate(seed):
            x_pred[0, i] = char_to_idx.get(c, 0)

        preds = model.predict(x_pred, verbose=0)[0]
        next_char = idx_to_char[np.argmax(preds)]

        generated += next_char
        seed = generated[-seq_len:]

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode()

    return jsonify({
        "ok": True,
        "generated": generated,
        "plot": f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;">'
    })


# ======================================================================
# 🔥 SAHAM
# ======================================================================
@app.route('/predict_stock')
def predict_stock_page():
    return render_template('saham.html')


@app.route('/api/predict_stock', methods=['POST'])
def predict_stock_api():
    data = request.json or {}
    symbol = data.get('symbol', 'BBCA')
    days = int(data.get('days', 7))

    csv_path = os.path.join(os.path.dirname(__file__), "Data Historis BBCA_Test2.csv")

    if not os.path.exists(csv_path):
        return jsonify({'ok': False, 'error': 'Dataset tidak ditemukan'}), 400

    df = pd.read_csv(csv_path)

    for col in ["Close", "Harga Penutupan", "close", "harga"]:
        if col in df.columns:
            price_col = col
            break
    else:
        return jsonify({'ok': False, 'error': 'Kolom harga tidak ditemukan'}), 400

    prices = df[price_col].dropna().tolist()
    actual = prices[-days:]

    preds = []
    for i in range(len(actual)):
        if i == 0:
            preds.append(actual[i])
        else:
            delta = actual[i] - actual[i - 1]
            preds.append(preds[-1] + delta * random.uniform(0.8, 1.2))

    labels = [f"Hari ke-{i+1}" for i in range(days)]

    return jsonify({
        "ok": True,
        "symbol": symbol,
        "model": "Simple Forecast",
        "days": days,
        "labels": labels,
        "actual": actual,
        "predictions": preds,
        "prediksi_terakhir": f"Rp {preds[-1]:,.0f}".replace(",", ".")
    })


# =======================================================
# 📸 HALAMAN YOLO
# =======================================================
@app.route('/object_detection')
def object_detection_page():
    return render_template('object_detection.html')


# =======================================================
# 🔥 API YOLO DETECTION
# =======================================================
@app.route('/api/object_detect', methods=['POST'])
def object_detect_api():
    if MODEL_YOLO is None:
        return jsonify({"ok": False, "msg": "Model YOLO tidak siap!"}), 500

    if 'image' not in request.files:
        return jsonify({'ok': False, 'msg': 'Tidak ada file gambar'}), 400

    file = request.files['image']
    img_bytes = file.read()

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"ok": False, "msg": "Gambar tidak bisa dibaca"}), 400

    results = MODEL_YOLO(img)
    detected_objects = []
    found = False

    for result in results:
        for box in result.boxes:
            found = True
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            conf = float(box.conf[0])
            cls = int(box.cls.item())
            label = MODEL_YOLO.names.get(cls, str(cls))

            detected_objects.append({
                "label": label,
                "confidence": round(conf * 100, 2)
            })

            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 200, 255), 3)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 200, 255), 2)

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode()

    return jsonify({
        "ok": True,
        "found": found,
        "objects": detected_objects,
        "image": img_base64
    })


# =======================================================
# 🚀 RUN SERVER
# =======================================================
if __name__ == "__main__":
    app.run(debug=True, port=8000)
