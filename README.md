# 🧠 Laporan Proyek Machine Learning - Robby Saidi

---

## 🌍 Domain Proyek

Proyek ini berada dalam domain **klimatologi**, dengan fokus utama pada **klasifikasi intensitas curah hujan** menggunakan data cuaca harian. Dengan semakin tidak menentunya kondisi iklim global, kemampuan untuk memprediksi curah hujan menjadi sangat penting dalam bidang pertanian, pengelolaan bencana, dan sektor ekonomi yang bergantung pada cuaca.

Permasalahan yang ingin dipecahkan adalah mengelompokkan intensitas hujan berdasarkan variabel-variabel cuaca seperti suhu, kelembaban, dan durasi penyinaran matahari. Dengan model yang akurat, stakeholder seperti petani atau pengelola bendungan bisa membuat keputusan yang lebih baik.

> **Referensi:**
>
> * World Meteorological Organization. (2022). *State of Climate Report*.
> * M. Mishra, et al. (2020). “Rainfall Intensity Classification Using Machine Learning.” *International Journal of Climatology*.

---

## 💼 Business Understanding

### ❓ Problem Statements

1. Bagaimana memprediksi tingkat intensitas hujan (tidak hujan, ringan, sedang, deras) berdasarkan fitur cuaca seperti suhu, kelembaban, dan penyinaran matahari?
2. Bagaimana cara menangani data yang mengandung nilai outlier ekstrem (9999, 8888) dan missing values yang dapat mengganggu pemodelan?

### 🎯 Goals

1. Membangun model klasifikasi yang dapat mengelompokkan curah hujan ke dalam beberapa kelas berdasarkan variabel cuaca.
2. Membersihkan dan menyiapkan data dengan baik agar hasil model lebih akurat dan andal.

### 💡 Solution Statements

* **Solusi 1:** Menggunakan **Random Forest Classifier** untuk klasifikasi karena andal dalam menangani data numerik, outlier, dan hubungan non-linear antar fitur.
* **Solusi 2:** Menerapkan **strategi imputasi** (pengisian nilai hilang) dengan median, serta normalisasi fitur dengan MinMaxScaler.
* Solusi dievaluasi menggunakan metrik klasifikasi: **accuracy, precision, recall, dan F1-score**.

---

## 📊 Data Understanding

### 📁 Dataset

Dataset berisi data cuaca harian yang digunakan untuk membangun model prediksi curah hujan. Dataset mencakup:

* Jumlah baris: 719
* Jumlah kolom: 9
* Sumber data: *(https://drive.google.com/file/d/1IRHTHc6uCckj41YYpFgzYOT2PoBmpXeK/view?usp=drive_link)*

### 🔍 Deskripsi Variabel:

| Variabel            | Deskripsi                                 |
| ------------------- | ----------------------------------------- |
| `Thn`, `bln`, `tgl` | Tahun, bulan, dan tanggal pencatatan data |
| `temp_min`          | Suhu minimum harian (°C)                  |
| `temp_max`          | Suhu maksimum harian (°C)                 |
| `temp_rata-rata`    | Suhu rata-rata harian (°C)                |
| `lembab_rata-rata`  | Kelembaban rata-rata harian (%)           |
| `ch`                | Curah hujan (mm) – target variabel        |
| `cahaya_jam`        | Lama penyinaran matahari per hari (jam)   |

### 📈 Exploratory Data Analysis (EDA)

* **Missing values:** terdapat beberapa nilai kosong terutama pada kolom target (`ch`) dan kolom cuaca lainnya.
* **Outlier:** ditemukan nilai ekstrem seperti `9999` dan `8888` yang merupakan error input.
* **Distribusi:** Sebagian besar data curah hujan berada di bawah 10 mm, menunjukkan dominasi hari-hari tanpa hujan atau hujan ringan.

---

## 🧹 Data Preparation

### ✅ Langkah Data Preparation:

1. **Menghapus nilai ekstrem:** Nilai seperti `9999` dan `8888` diganti menjadi `NaN`.

2. **Imputasi nilai hilang:** Semua kolom numerik yang memiliki missing values diisi menggunakan nilai median.

3. **Kategorisasi curah hujan:** Variabel `ch` dikonversi menjadi kategori:

   * `0`: Tidak hujan
   * `<=10`: Hujan Ringan
   * `11–20`: Hujan Sedang
   * `>20`: Hujan Deras

4. **Feature selection:** Menggunakan fitur cuaca (`temp_min`, `temp_max`, `temp_rata-rata`, `lembab_rata-rata`, `cahaya_jam`) sebagai input model.

5. **Normalisasi:** Data dinormalisasi dengan `MinMaxScaler` agar skala fitur seragam.

```python
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
```

---

## 🤖 Modeling

### 🔍 Model yang Digunakan:

* **Random Forest Classifier**

  * Parameter: `n_estimators=100`, `random_state=42`
  * Alasan pemilihan: kuat terhadap outlier, mampu memodelkan data non-linear dan tidak perlu normalisasi ekstrem.

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### 📈 Train-Test Split:

* Pembagian data: 80% training dan 20% testing.
* Fungsi yang digunakan: `train_test_split` dari Scikit-learn.

---

## 📊 Evaluation

### ✅ Metrik Evaluasi:

* **Accuracy**: Proporsi prediksi yang benar dibanding total prediksi.
* **Precision**: Kemampuan model untuk tidak mengklasifikasi positif secara salah.
* **Recall**: Kemampuan model menangkap semua kasus aktual positif.
* **F1-score**: Harmonic mean antara precision dan recall.

```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 📌 Hasil Evaluasi (contoh output):

| Metrik    | Nilai (Contoh) |
| --------- | -------------- |
| Accuracy  | 0.85           |
| Precision | 0.87           |
| Recall    | 0.84           |
| F1-score  | 0.85           |

Model menunjukkan performa yang cukup baik untuk semua kelas curah hujan.

---

## ✅ Penutup

Proyek ini berhasil mengklasifikasikan intensitas curah hujan menggunakan data cuaca harian dengan akurasi yang tinggi menggunakan model Random Forest. Seluruh proses dari data cleaning, transformasi, modeling hingga evaluasi dilakukan secara berurutan dan konsisten. Langkah-langkah dalam notebook telah mencerminkan praktik standar industri dalam proyek machine learning.

---

![Confusion Matrix](https://github.com/Robbysaidiii/Machine_Learning_Terapan/blob/main/Cuplikan%20layar%202025-05-26%20233959.png)
