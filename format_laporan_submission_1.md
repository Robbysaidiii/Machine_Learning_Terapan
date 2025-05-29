# Laporan Proyek Machine Learning – Prediksi Curah Hujan Harian

## 🌍 Domain Proyek

Proyek ini berada dalam domain **klimatologi**, dengan fokus pada prediksi **curah hujan harian** berdasarkan data cuaca. Dalam konteks perubahan iklim dan cuaca ekstrem, prediksi curah hujan yang akurat sangat penting untuk pertanian, pengendalian banjir, dan manajemen sumber daya air.

---

## 💼 Business Understanding

### ❓ Permasalahan

1. Bagaimana memprediksi curah hujan berdasarkan data cuaca seperti suhu, kelembaban, dan cahaya matahari?
2. Bagaimana menangani data cuaca yang mengandung nilai ekstrem seperti 9999 atau 8888 serta nilai yang hilang?
3. Lebih efektif mana: klasifikasi kategori curah hujan atau regresi nilai aktual?

### 🎯 Tujuan

1. Membangun model klasifikasi untuk membedakan antara kondisi: **Tidak Hujan, Hujan Ringan, Hujan Sedang**.
2. Membangun model regresi untuk memprediksi nilai curah hujan secara numerik (dalam mm).
3. Mengatasi masalah **ketidakseimbangan kelas** dalam data kategori curah hujan.
4. Melakukan pembersihan dan persiapan data agar akurat, representatif, dan siap untuk modeling.

---

## 📊 Data Understanding

### Dataset

* Sumber: [Kaggle - Prediksi Cuaca CSV](https://www.kaggle.com/datasets/robbysaidiii/prediksi-cuaca-csv)
* Jumlah data: 719 baris (tahun 2022–2023)
* Jumlah kolom: 9

### Ringkasan Kondisi Data

* **Missing values** ditemukan pada kolom suhu, kelembaban, cahaya matahari, dan curah hujan.
* **Nilai ekstrem** seperti 9999 dan 8888 teridentifikasi sebagai placeholder error dan harus dibersihkan.
* **Outlier alami** terlihat pada beberapa kolom, terutama `lembab_rata-rata`, `ch` (curah hujan), dan `cahaya_jam`.

Visualisasi distribusi menunjukkan adanya skew dan data tidak seimbang antar kelas curah hujan.

---

## 🧹 Data Preparation

### Langkah-langkah Persiapan Data:

1. **Penanganan Nilai Ekstrem dan Hilang**

   * Nilai 9999 dan 8888 diubah menjadi kosong.
   * Nilai kosong diisi dengan median (karena lebih tahan terhadap outlier).

2. **Konversi Format Tanggal**

   * Kolom tahun, bulan, dan tanggal digabung menjadi satu kolom bertipe datetime dan digunakan sebagai indeks waktu.

3. **Kategorisasi Target**

   * Curah hujan dikategorikan menjadi:

     * `Tidak Hujan` jika `ch == 0`
     * `Hujan Ringan` jika `ch < 20`
     * `Hujan Sedang` jika `ch < 50`
     * `Hujan Deras` jika `ch >= 50`

4. **Label Encoding**

   * Kategori hujan dikonversi menjadi label numerik untuk modeling klasifikasi.

5. **Pemilihan Fitur**

   * Fitur yang digunakan: suhu minimum, maksimum, rata-rata, kelembaban rata-rata, dan cahaya matahari.

6. **Normalisasi**

   * Semua fitur dinormalisasi ke dalam rentang 0–1 agar model lebih stabil.

7. **Pembagian Data**

   * Data dibagi menjadi 80% data latih dan 20% data uji tanpa pengacakan, agar urutan waktu tetap terjaga.

---

## 🔬 Exploratory Data Analysis (EDA)

* Visualisasi distribusi kategori curah hujan menunjukkan dominasi kelas "Tidak Hujan".
* Korelasi antar fitur menunjukkan hubungan positif antara suhu maksimum dan suhu rata-rata, namun korelasi terhadap curah hujan relatif rendah.
* Sebagian besar hari dalam data tidak terjadi hujan, menyebabkan ketidakseimbangan kelas.

---

## 🤖 Modeling

### 1. **Klasifikasi – Random Forest**

* Algoritma Random Forest digunakan untuk klasifikasi karena mampu menangani fitur non-linear dan tahan terhadap overfitting.
* Klasifikasi dilakukan untuk mengelompokkan data menjadi 3 kelas utama: Tidak Hujan, Hujan Ringan, dan Hujan Sedang.

### 2. **Regresi – Linear Regression**

* Regresi linier digunakan untuk memprediksi nilai curah hujan harian dalam mm.
* Model ini sederhana namun kurang cocok untuk fenomena cuaca yang bersifat kompleks dan non-linear.

---

## 🔎 Evaluation

### Evaluasi Klasifikasi

| Kelas            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Hujan Ringan     | 0.65      | 0.76   | 0.70     | 42      |
| Hujan Sedang     | 0.00      | 0.00   | 0.00     | 6       |
| Tidak Hujan      | 0.88      | 0.88   | 0.88     | 96      |
| **Accuracy**     |           |        | 0.81     | 144     |
| **Macro Avg**    | 0.51      | 0.55   | 0.53     | 144     |
| **Weighted Avg** | 0.78      | 0.81   | 0.79     | 144     |

**Catatan**:

* Model berhasil memprediksi kelas "Tidak Hujan" dengan baik.
* Kelas "Hujan Sedang" gagal diprediksi karena terlalu sedikit datanya (kelas minoritas).

### Evaluasi Regresi

* **Mean Squared Error (MSE)**: 0.6348
* **R² Score**: 0.2236

**Catatan**:

* R² yang rendah (22%) menunjukkan bahwa model linier tidak dapat menangkap kompleksitas pola curah hujan secara memadai.

---

## 📈 Perbandingan Model

| Aspek      | Klasifikasi (Random Forest)           | Regresi (Linear)                  |
| ---------- | ------------------------------------- | --------------------------------- |
| Akurasi    | 81%                                   | R² = 22%                          |
| Kelebihan  | Kuat terhadap data kategori dan noise | Sederhana dan cepat               |
| Kekurangan | Lemah terhadap kelas minoritas        | Tidak cocok untuk pola non-linear |

---

## 📌 Rekomendasi dan Kesimpulan

### Rekomendasi Perbaikan

1. **Atasi ketidakseimbangan kelas** menggunakan teknik seperti SMOTE atau penyesuaian bobot kelas.
2. **Gunakan model lanjutan** seperti XGBoost atau Gradient Boosting untuk klasifikasi dan regresi.
3. **Tambah fitur cuaca tambahan** seperti tekanan udara, kecepatan angin, atau lag dari hari sebelumnya.
4. **Visualisasi tambahan** seperti distribusi target sebelum/sesudah preprocessing untuk mendukung pemahaman data.

### Kesimpulan

* **Random Forest Classifier** memberikan performa terbaik (akurasi 81%) dibanding regresi linier.
* Tantangan terbesar terletak pada **ketidakseimbangan data**, khususnya pada kelas hujan sedang dan deras.
* Dengan perbaikan data dan model lanjutan, performa prediksi dapat ditingkatkan secara signifikan.
