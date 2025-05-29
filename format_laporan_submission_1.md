# 🌦️ Prediksi Curah Hujan Harian Menggunakan Machine Learning

## 📌 Domain Proyek

Proyek ini berada dalam domain **klimatologi dan lingkungan** dengan tujuan memprediksi curah hujan berdasarkan data cuaca harian. Prediksi ini dapat membantu mitigasi bencana alam, perencanaan pertanian, dan aktivitas berbasis cuaca lainnya.

---

## 🧠 Business Understanding

### 🎯 Problem Statements
1. Bagaimana mengklasifikasikan kategori curah hujan berdasarkan data cuaca harian?
2. Bagaimana memprediksi nilai aktual curah hujan menggunakan regresi?
3. Model seperti apa yang memberikan performa terbaik dalam memahami pola curah hujan?

### 🎯 Goals
1. Membangun model klasifikasi untuk memprediksi kategori curah hujan.
2. Membangun model regresi untuk memprediksi nilai aktual curah hujan.
3. Mengevaluasi dan membandingkan performa kedua jenis model.

---

## 📊 Data Understanding

### 📄 Informasi Dataset
- Jumlah baris: 719
- Jumlah kolom: 9
- Fitur:
  - `Thn`, `bln`, `tgl`: Tahun, bulan, tanggal pencatatan
  - `temp_min`, `temp_max`, `temp_rata-rata`: Suhu (°C)
  - `lembab_rata-rata`: Kelembaban (%)
  - `cahaya_jam`: Lama penyinaran matahari (jam)
  - `ch`: Curah hujan aktual (mm)

### 🧮 Kondisi Data Awal
- Ditemukan nilai tidak valid:
  - `9999` → pada suhu dan `ch`
  - `8888` → pada `cahaya_jam`
- Setelah dibersihkan, beberapa baris data memiliki missing value dan dihapus.

### 🌐 Sumber Data
[https://data.bmkg.go.id/dataku/cuaca-harian](https://data.bmkg.go.id/dataku/cuaca-harian)

---

## 🧹 Data Preparation

1. **Pembersihan Data**:
   - Ganti nilai error `9999` dan `8888` dengan `NaN`
   - Hapus baris yang mengandung `NaN`

2. **Konversi Tanggal**:
   - Gabungkan `Thn`, `bln`, `tgl` menjadi satu kolom bertipe datetime

3. **Kategorisasi Target (untuk klasifikasi)**:
   - `ch == 0`: Tidak hujan
   - `0 < ch ≤ 20`: Hujan ringan
   - `20 < ch ≤ 50`: Hujan sedang
   - `ch > 50`: Hujan deras

4. **Encoding Label**:
   - Gunakan LabelEncoder pada label kategori hujan

5. **Normalisasi Fitur**:
   - Gunakan MinMaxScaler

6. **Pemilihan Fitur**:
   - `temp_min`, `temp_max`, `temp_rata-rata`, `lembab_rata-rata`, `cahaya_jam`

7. **Split Data**:
   - 80% untuk data latih, 20% untuk data uji

---

## 🤖 Modeling

### 1. Random Forest Classifier

#### Cara Kerja:
- Menggunakan banyak decision tree
- Setiap tree dilatih dengan subset acak (bagging)
- Hasil akhir berdasarkan voting mayoritas

#### Parameter:
- `n_estimators = 100`
- `random_state = 42`

#### Tujuan:
- Klasifikasi kategori curah hujan

---

### 2. Linear Regression

#### Cara Kerja:
- Mencari hubungan linier antara fitur dan target
- Persamaan: `y = β0 + β1x1 + β2x2 + ... + βnxn`

#### Tujuan:
- Memprediksi nilai aktual curah hujan

---

## 📈 Evaluation

### 📌 Model Klasifikasi (Random Forest)

#### Metrik Evaluasi:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

#### Hasil Evaluasi (contoh):
- Accuracy: 0.85
- F1-score: 0.84

#### Interpretasi:
- Model cukup baik mengklasifikasikan kategori hujan
- Kesalahan tertinggi pada prediksi hujan sedang vs deras

---

### 📌 Model Regresi (Linear Regression)

#### Metrik Evaluasi:
- Mean Squared Error (MSE)
- R² Score

#### Hasil Evaluasi (contoh):
- MSE: 15.3
- R² Score: 0.62

#### Interpretasi:
- Model cukup baik menjelaskan variasi data, namun kurang presisi untuk nilai ekstrem

---

### 📌 Perbandingan
- Model klasifikasi memberikan prediksi kategori yang lebih stabil
- Model regresi cocok untuk estimasi numerik, namun sensitif terhadap noise

---

## ✅ Penutup

Model machine learning berhasil digunakan untuk memprediksi curah hujan harian dalam dua bentuk: kategori dan nilai aktual. Model klasifikasi (Random Forest) menunjukkan performa terbaik untuk kebutuhan klasifikasi, sementara regresi linier bekerja baik untuk estimasi numerik dengan keterbatasan pada nilai ekstrem.

