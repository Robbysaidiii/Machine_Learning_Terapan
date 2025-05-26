Berikut adalah **penjelasan lengkap dan terstruktur** dari hasil proyek Machine Learning Anda, **menggabungkan pendekatan klasifikasi dan regresi**, sesuai dengan isi notebook dan evaluasi model yang telah dilakukan:

---

# 🧠 Laporan Proyek Machine Learning – Robby Saidi Prasetyo

---

## 🌍 Domain Proyek

Proyek ini berada dalam domain **klimatologi**, yang berfokus pada **prediksi curah hujan harian** berdasarkan data cuaca. Dalam era perubahan iklim global yang semakin tidak menentu, prediksi cuaca yang akurat sangat penting bagi sektor pertanian, pengelolaan bencana, dan perencanaan infrastruktur.

---

## 💼 Business Understanding

### ❓ Problem Statements

1. Bagaimana memprediksi **tingkat curah hujan** berdasarkan fitur cuaca seperti suhu, kelembaban, dan durasi penyinaran matahari?
2. Bagaimana mengelola data yang mengandung **nilai ekstrem (9999, 8888)** dan **nilai hilang**?
3. Model seperti apa yang lebih efektif: **klasifikasi kategori curah hujan**, atau **prediksi nilai curah hujan aktual**?

### 🎯 Goals

* Membangun dua jenis model:

  * **Model Klasifikasi**: Mengelompokkan curah hujan ke dalam kelas: *Tidak Hujan*, *Hujan Ringan*, *Hujan Sedang*.
  * **Model Regresi**: Memprediksi nilai numerik dari curah hujan harian (dalam mm).
* Membersihkan dan menyiapkan data agar representatif dan tidak bias.

---

## 📊 Data Understanding

### Dataset

* Sumber: Dataset cuaca harian
* Jumlah baris: 719
* Jumlah kolom: 9
Jumlah Baris dan Kolom:

* Dataset memiliki 719 baris dan 9 kolom.
Artinya data ini merupakan hasil pengamatan harian selama 719 hari, yaitu dari tahun 2022 hingga 2023.
Tipe Data Kolom:

* Terdapat 2 tipe data:

float64 sebanyak 6 kolom, digunakan untuk data numerik pecahan/desimal (misalnya suhu, kelembaban, cahaya).
int64 sebanyak 3 kolom, digunakan untuk angka bulat seperti tahun (Thn), bulan (bln), dan tanggal (tgl).

### Variabel

| Kolom               | Deskripsi                       |
| ------------------- | ------------------------------- |
| `Thn`, `bln`, `tgl` | Tanggal pencatatan              |
| `temp_min`          | Suhu minimum harian (°C)        |
| `temp_max`          | Suhu maksimum harian (°C)       |
| `temp_rata-rata`    | Suhu rata-rata harian (°C)      |
| `lembab_rata-rata`  | Kelembaban rata-rata harian (%) |
| `ch`                | Curah hujan (mm) – **target**   |
| `cahaya_jam`        | Lama penyinaran matahari (jam)  |

---

## 🧹 Data Preparation

1. **Menghapus nilai ekstrem** seperti `9999` dan `8888`, digantikan dengan `NaN`.
2. **Imputasi** nilai hilang dengan median (tahan terhadap outlier).
3. **Kategorisasi target** (`ch`) untuk klasifikasi:

   * `0`: Tidak hujan
   * `<=10`: Hujan ringan
   * `11–20`: Hujan sedang
4. **Normalisasi fitur** dengan `MinMaxScaler`.

```python
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
```

---

## 🤖 Modeling

### 1. **Klasifikasi – Random Forest Classifier**

* Model: `RandomForestClassifier(n_estimators=100, random_state=42)`
* Pembagian data: 80% training – 20% testing

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

#### 🔎 Hasil Evaluasi Klasifikasi:

```text
              precision    recall  f1-score   support

hujan ringan       0.65      0.76      0.70        42
hujan sedang       0.00      0.00      0.00         6
 tidak hujan       0.88      0.88      0.88        96

    accuracy                           0.81       144
   macro avg       0.51      0.55      0.53       144
weighted avg       0.78      0.81      0.79       144
```

#### ✅ Interpretasi:

* **Akurasi keseluruhan: 81%** → sangat baik untuk klasifikasi dasar.
* Performa bagus untuk kelas dominan (*tidak hujan*), cukup baik untuk *hujan ringan*.
* **Gagal mengklasifikasikan "hujan sedang"** (recall dan precision = 0) karena:

  * Data sangat sedikit (support = 6)
  * Fitur tumpang tindih (overlap antar kelas)
* Macro average F1 rendah (0.53) → menunjukkan **ketimpangan antar kelas**.

---

### 2. **Regresi – Linear Regression**

* Model: `LinearRegression()`
* Target: `ch` (nilai curah hujan aktual dalam mm)

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

#### 🔎 Hasil Evaluasi Regresi:

| Metrik             | Nilai  |
| ------------------ | ------ |
| Mean Squared Error | 0.6348 |
| R² Score           | 0.2236 |

#### ✅ Interpretasi:

* **R² Score = 0.22** → hanya menjelaskan 22% variasi dalam target.
* MSE rendah, tetapi karena R² rendah, ini menunjukkan model **tidak dapat menangkap kompleksitas hubungan fitur dengan curah hujan**.
* Model linier terlalu sederhana untuk fenomena iklim yang **non-linear**.

---

## 🔄 Perbandingan Klasifikasi vs Regresi

| Aspek           | Klasifikasi (RF)                          | Regresi (Linear)                  |
| --------------- | ----------------------------------------- | --------------------------------- |
| Tipe Target     | Kategori                                  | Numerik                           |
| Performa        | Baik (accuracy = 81%)                     | Lemah (R² = 22%)                  |
| Kelas Minoritas | Sulit diklasifikasi                       | Tidak ditangani khusus            |
| Kelebihan       | Robust, cocok untuk data kategori         | Sederhana, cepat                  |
| Kekurangan      | Sensitif terhadap jumlah data kelas kecil | Tidak bisa modelkan pola kompleks |

---

## 📌 Rekomendasi

1. **Untuk Klasifikasi**:

   * Gunakan **resampling (SMOTE)** atau **class\_weight='balanced'**.
   * Coba model boosting seperti **XGBoost**, **LightGBM**, atau **GradientBoostingClassifier**.
   * Lakukan **grid search** untuk tuning parameter.

2. **Untuk Regresi**:

   * Ganti ke **RandomForestRegressor** atau **GradientBoostingRegressor**.
   * Gunakan **log-transformasi** pada target jika distribusinya skewed.
   * Coba **non-linear model** yang lebih kompleks.

---

## ✅ Penutup

Proyek ini membuktikan bahwa pendekatan **klasifikasi** saat ini memberikan hasil yang **lebih baik** dibandingkan regresi dalam memodelkan curah hujan harian. Model Random Forest menunjukkan performa yang baik untuk dua dari tiga kelas, sementara regresi linier masih terlalu sederhana untuk menjelaskan variabilitas curah hujan.

Langkah selanjutnya adalah:

* **Memperkuat klasifikasi** dengan penanganan kelas imbang dan eksplorasi model lain.
* **Meningkatkan model regresi** dengan pendekatan non-linear.

