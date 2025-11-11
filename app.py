from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ==============================
# 🏠 ROUTE HALAMAN UTAMA
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

    if gate == "AND":
        result = a & b
    elif gate == "OR":
        result = a | b
    elif gate == "XOR":
        result = a ^ b
    elif gate == "NAND":
        result = int(not (a & b))
    elif gate == "NOR":
        result = int(not (a | b))
    elif gate == "XNOR":
        result = int(not (a ^ b))
    else:
        return jsonify({'ok': False, 'error': 'Invalid Gate'}), 400

    # Tabel kebenaran
    truth_table = []
    for x in [0, 1]:
        for y in [0, 1]:
            if gate == "AND":
                val = x & y
            elif gate == "OR":
                val = x | y
            elif gate == "XOR":
                val = x ^ y
            elif gate == "NAND":
                val = int(not (x & y))
            elif gate == "NOR":
                val = int(not (x | y))
            elif gate == "XNOR":
                val = int(not (x ^ y))
            truth_table.append({'A': x, 'B': y, 'Output': val})

    return jsonify({'ok': True, 'result': result, 'truth_table': truth_table})


# ==============================
# 🔹 GENERATE TEKS (AI TEKS)
# ==============================
@app.route('/predict_word')
def predict_word_page():
    return render_template('generate.html')

@app.route('/api/predict_word', methods=['POST'])
def predict_word_api():
    data = request.json or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'ok': False, 'error': 'Teks tidak boleh kosong'}), 400

    last_word = text.split()[-1]
    predicted = last_word + "_next"

    return jsonify({'ok': True, 'input': text, 'prediction': predicted})


# ==============================
# 🔹 PREDIKSI SAHAM
# ==============================
@app.route('/predict_stock')
def predict_stock_page():
    return render_template('saham.html')

@app.route('/api/predict_stock', methods=['POST'])
def predict_stock_api():
    data = request.json or {}
    symbol = data.get('symbol', 'DUMMY').upper()
    base = data.get('base', 100.0)

    try:
        base = float(base)
    except:
        return jsonify({'ok': False, 'error': 'Harga dasar harus berupa angka'}), 400

    # Simulasi prediksi 5 hari (dummy)
    preds = [round(base * (1 + i * 0.015), 2) for i in range(1, 6)]
    days = [f"Hari ke-{i}" for i in range(1, 6)]

    return jsonify({'ok': True, 'symbol': symbol, 'days': days, 'predictions': preds})


# ==============================
# 🚀 RUN APP
# ==============================
if __name__ == '__main__':
    app.run(debug=True)
