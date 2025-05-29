# Laporan Proyek Machine Learning – Prediksi Curah Hujan Harian

## 🌍 Domain Proyek

Proyek ini berada dalam domain klimatologi, yang berfokus pada prediksi curah hujan harian berdasarkan data cuaca. Dalam era perubahan iklim global yang semakin tidak menentu, prediksi cuaca yang akurat sangat penting bagi sektor pertanian, pengelolaan bencana, dan perencanaan infrastruktur.

## 💼 Business Understanding

### ❓ Problem Statements

1. Bagaimana memprediksi tingkat curah hujan berdasarkan fitur cuaca seperti suhu, kelembaban, dan durasi penyinaran matahari?
2. Bagaimana mengelola data yang mengandung nilai ekstrem (9999, 8888) dan nilai hilang?
3. Model seperti apa yang lebih efektif: klasifikasi kategori curah hujan, atau prediksi nilai curah hujan aktual?

### 🎯 Goals

1. Membangun Model Klasifikasi untuk mengelompokkan curah hujan ke dalam kelas: Tidak Hujan, Hujan Ringan, Hujan Sedang
2. Membangun Model Regresi untuk memprediksi nilai numerik dari curah hujan harian (dalam mm)
3. Mengembangkan strategi handling data tidak seimbang untuk kelas minoritas
4. Membersihkan dan menyiapkan data agar representatif dan tidak bias

---

## 📊 Data Understanding

### Dataset

Sumber: [Dataset Cuaca Harian dari kaggel](https://www.kaggle.com/datasets/robbysaidiii/prediksi-cuaca-csv) 

Dataset ini berisi data cuaca harian yang terdiri dari:

* Jumlah baris: 719 (pengamatan harian dari tahun 2022-2023)
* Jumlah kolom: 9

### Kondisi Data Awal

* Missing values:

  * temp\_min: 2 nilai hilang
  * temp\_max: 5 nilai hilang
  * temp\_rata-rata: 3 nilai hilang
  * lembab\_rata-rata: 3 nilai hilang
  * cahaya\_jam: 4 nilai hilang
  * ch: 83 nilai hilang
* Outliers:

  * Nilai ekstrem 9999 dan 8888 ditemukan pada beberapa kolom
  * Outlier alami ditemukan pada kolom lembab\_rata-rata (59), ch (26), dan cahaya\_jam (38)

### Variabel

| Kolom             | Deskripsi                       |
| ----------------- | ------------------------------- |
| Thn, bln, tgl     | Tanggal pencatatan              |
| temp\_min         | Suhu minimum harian (°C)        |
| temp\_max         | Suhu maksimum harian (°C)       |
| temp\_rata-rata   | Suhu rata-rata harian (°C)      |
| lembab\_rata-rata | Kelembaban rata-rata harian (%) |
| ch                | Curah hujan (mm) – target       |
| cahaya\_jam       | Lama penyinaran matahari (jam)  |

---

## 🧹 Data Preparation

### Tahapan Persiapan Data:

1. **Penanganan Nilai Ekstrem**:

   * Mengganti nilai 9999 dan 8888 dengan NaN
   * Imputasi nilai hilang dengan median (robust terhadap outlier)

2. **Konversi Format Tanggal**:

   * Menggabungkan kolom Thn, bln, tgl menjadi satu kolom datetime
   * Menjadikan kolom tanggal sebagai index

3. **Kategorisasi Target**:

   * ch == 0: tidak hujan
   * ch < 20: hujan ringan
   * ch < 50: hujan sedang
   * ch >= 50: hujan deras

4. **Encoding Label**:

   * Menggunakan LabelEncoder untuk mengubah kategori menjadi nilai numerik

5. **Pemilihan Fitur**:

   * Fitur yang digunakan: \['temp\_min', 'temp\_max', 'temp\_rata-rata', 'lembab\_rata-rata', 'cahaya\_jam']
   * Target: 'label' (hasil encoding kategori hujan)

6. **Normalisasi Data**:

   * Menggunakan MinMaxScaler untuk menormalkan fitur ke rentang \[0,1]

7. **Pembagian Data**:

   * Train-test split 80%-20% tanpa shuffle untuk menjaga urutan waktu

---

## 🤖 Modeling

### 1. Klasifikasi – Random Forest Classifier

**Penjelasan Algoritma**:
Random Forest adalah metode ensemble learning yang membangun banyak pohon keputusan dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting. Setiap pohon dilatih pada subset data dan fitur yang berbeda, kemudian hasil prediksi ditentukan melalui voting mayoritas.

**Keunggulan**:

* Robust terhadap overfitting
* Dapat menangani fitur non-linear dan data dengan noise

**Implementasi**:

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

**Parameter**:

* n\_estimators=100: Jumlah pohon dalam forest
* random\_state=42: Untuk reproduktibilitas hasil

---

### 2. Regresi – Linear Regression

**Penjelasan Algoritma**:
Linear Regression memodelkan hubungan linear antara variabel independen (fitur) dan dependen (target) dengan mencari garis lurus yang paling sesuai dengan data.

**Implementasi**:

```python
model_reg = LinearRegression()
model_reg.fit(X_train, y_train)
```

---

## 🔎 Evaluation

### Hasil Evaluasi Klasifikasi (Random Forest):

| Kelas            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Hujan Ringan     | 0.65      | 0.76   | 0.70     | 42      |
| Hujan Sedang     | 0.00      | 0.00   | 0.00     | 6       |
| Tidak Hujan      | 0.88      | 0.88   | 0.88     | 96      |
| **Accuracy**     |           |        | 0.81     | 144     |
| **Macro Avg**    | 0.51      | 0.55   | 0.53     | 144     |
| **Weighted Avg** | 0.78      | 0.81   | 0.79     | 144     |

**Interpretasi**:

* Akurasi keseluruhan 81% menunjukkan model cukup baik untuk klasifikasi dasar
* Performa bagus untuk kelas dominan (tidak hujan)
* Gagal mengklasifikasikan "hujan sedang" karena ketidakseimbangan data (hanya 6 sampel)

---

### Hasil Evaluasi Regresi (Linear Regression):

* Mean Squared Error: 0.6348
* R² Score: 0.2236

**Interpretasi**:

* R² Score = 0.22 → hanya menjelaskan 22% variasi dalam target
* Model linier terlalu sederhana untuk fenomena iklim yang kompleks

---

### Perbandingan Model:

| Aspek      | Klasifikasi (RF)                  | Regresi (Linear)                   |
| ---------- | --------------------------------- | ---------------------------------- |
| Akurasi    | 81%                               | 22% (R² Score)                     |
| Kelebihan  | Robust, cocok untuk data kategori | Sederhana, cepat                   |
| Kekurangan | Sensitif terhadap imbalance kelas | Tidak bisa menangkap pola kompleks |

---

## 📌 Rekomendasi dan Kesimpulan

### Rekomendasi Perbaikan:

1. Untuk Klasifikasi:

   * Gunakan teknik handling imbalance class seperti SMOTE atau class weighting
   * Coba model boosting seperti XGBoost atau Gradient Boosting
   * Lakukan hyperparameter tuning untuk optimasi model

2. Untuk Regresi:

   * Gunakan model non-linear seperti Random Forest Regressor atau XGBoost Regressor
   * Pertimbangkan transformasi target jika distribusi target skewed
   * Tambahkan teknik feature engineering untuk meningkatkan representasi fitur

3. Tambahkan visualisasi distribusi target sebelum dan sesudah preprocessing untuk memperjelas kondisi data

### Kesimpulan:

Pendekatan klasifikasi dengan Random Forest memberikan hasil yang lebih baik (akurasi 81%) dibanding regresi linier untuk prediksi curah hujan. Namun, model masih kesulitan memprediksi kelas minoritas (hujan sedang) karena ketidakseimbangan data.

Langkah selanjutnya adalah memperbaiki model dengan teknik handling imbalance class dan mencoba algoritma yang lebih advanced untuk meningkatkan performa prediksi.

---

