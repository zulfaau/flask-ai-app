# Tugas Mata Kuliah Deep Learning - Template Web + Flask API (Mock)

Struktur proyek sederhana untuk memenuhi tugas web yang diinstruksikan:

- Frontend: HTML/CSS/JS (di folder templates/ dan static/)
- Backend: Flask (app.py) menyediakan 3 API sederhana:
  - `/api/calc` : calculator operator logika (AND/OR/XOR)
  - `/api/predict_word` : mock prediksi kata berikutnya
  - `/api/predict_stock` : mock prediksi harga saham (5 hari)

## Cara menjalankan (lokal)
1. Buat virtualenv (opsional) dan aktifkan.
2. Install requirement: `pip install -r requirements.txt`
3. Jalankan server: `python app.py`
4. Buka browser di `http://127.0.0.1:5000/`

Kamu bisa mengganti bagian mock dengan model ML nyata di Python dan menghubungkannya melalui endpoint yang sama.
