import sklearn
from flask import Flask, render_template, request
from model import load, prediksi

app = Flask(__name__)

# Load model dan scaler
load()

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    # Menangkap data yang diinput user melalui form
    nama = request.form['nama']
    asal_kelas = request.form['asal_kelas']
    lama_bekerja = int(request.form['lama_bekerja'])
    jam_kerja_per_bulan = int(request.form['jam_kerja_perbulan'])
    kecelakaan_kerja = int(request.form['is_pernah_kecelakaan_kerja'])
    gaji = int(request.form['kategori_gaji'])
    tingkat_kepuasan = int(request.form['tingkat_kepuasan']) / 100

    # Melakukan prediksi menggunakan model yang telah dibuat
    data = [[tingkat_kepuasan, lama_bekerja, kecelakaan_kerja, gaji, jam_kerja_per_bulan]]
    prediction_result, confidence = prediksi(data)

    # Mengirimkan variabel hasil prediksi dan data user ke template
    return render_template(
        'index.html',
        hasil_prediksi=prediction_result,
        nilai_kepercayaan=confidence,
        nama=nama,
        asal_kelas=asal_kelas
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
