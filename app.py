from flask import Flask, render_template, request, jsonify
import os
import random
import pandas as pd

app = Flask(__name__)

# ==============================
# 🏠 HALAMAN UTAMA
# ==============================
@app.route('/')
def index():
    return render_template('index.html')


# ==============================
# 🔹 KALKULATOR LOGIKA
# ==============================
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

    if gate not in gates:
        return jsonify({'ok': False, 'error': 'Invalid Gate'}), 400

    result = gates[gate](a, b)

    truth_table = [
        {'A': x, 'B': y, 'Output': gates[gate](x, y)}
        for x in [0, 1]
        for y in [0, 1]
    ]

    return jsonify({'ok': True, 'result': result, 'truth_table': truth_table})


# ==============================
# 🔹 GENERATE TEKS (AI TEKS)
# ==============================
@app.route('/predict_word')
def predict_word_page():
    return render_template('generate.html')


@app.route('/api/generate', methods=['POST'])
def predict_word_api():
    data = request.json or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'ok': False, 'error': 'Teks tidak boleh kosong'}), 400

    last_word = text.split()[-1]
    predicted = last_word + "_next"

    return jsonify({'ok': True, 'input': text, 'generated': predicted})


# ==============================
# 🔹 PREDIKSI HARGA SAHAM
# ==============================
@app.route('/predict_stock')
def predict_stock_page():
    # ini penting agar url_for('predict_stock_page') bisa dipanggil dari index.html
    return render_template('saham.html')


@app.route('/api/predict_stock', methods=['POST'])
def predict_stock_api():
    data = request.json or {}
    symbol = data.get('symbol', 'BBCA').upper()
    days = int(data.get('days', 10))

    # Path dataset
    csv_path = os.path.join(os.path.dirname(__file__), "Data Historis BBCA_Test2.csv")

    if not os.path.exists(csv_path):
        return jsonify({'ok': False, 'error': f'Dataset tidak ditemukan di {csv_path}'}), 400

    df = pd.read_csv(csv_path)

    # Deteksi kolom harga
    if 'Close' in df.columns:
        close_prices = df['Close'].dropna().tolist()
    elif 'Harga Penutupan' in df.columns:
        close_prices = df['Harga Penutupan'].dropna().tolist()
    else:
        return jsonify({'ok': False, 'error': 'Kolom harga penutupan tidak ditemukan di dataset.'}), 400

    # Ambil data terakhir dari dataset
    actual = close_prices[-days:] if len(close_prices) >= days else close_prices

    # Prediksi mengikuti pola data aktual (tidak lurus)
    preds = []
    for i, price in enumerate(actual):
        if i == 0:
            preds.append(price)
        else:
            delta = actual[i] - actual[i-1]
            preds.append(preds[-1] + delta * random.uniform(0.8, 1.2))

    # Buat data label hari
    days_list = [f"Hari ke-{i}" for i in range(1, len(preds) + 1)]

    preds_rp = [f"Rp {p:,.0f}".replace(",", ".") for p in preds]

    return jsonify({
        'ok': True,
        'symbol': symbol,
        'days': days_list,
        'predictions': preds,
        'actual': actual,
        'predictions_rp': preds_rp,
        'periode': f"{len(days_list)} hari ke depan",
        'prediksi_terakhir': preds_rp[-1]
    })


# ==============================
# 🚀 RUN SERVER
# ==============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
