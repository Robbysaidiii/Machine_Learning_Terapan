# Laporan Proyek Machine Learning – Prediksi Curah Hujan Harian

## 🌍 Domain Proyek

Proyek ini berada dalam domain klimatologi, yang berfokus pada prediksi curah hujan harian berdasarkan data cuaca. Dalam era perubahan iklim global yang semakin tidak menentu, prediksi cuaca yang akurat sangat penting bagi sektor pertanian, pengelolaan bencana, dan perencanaan infrastruktur.

---

## 💼 Business Understanding

### ❓ Problem Statements

1. Bagaimana memprediksi tingkat curah hujan berdasarkan fitur cuaca seperti suhu, kelembaban, dan durasi penyinaran matahari?
2. Bagaimana mengelola data yang mengandung nilai ekstrem (9999, 8888) dan nilai hilang?
3. Model seperti apa yang lebih efektif: klasifikasi kategori curah hujan, atau prediksi nilai curah hujan aktual?

### 🎯 Goals

1. Membangun Model Klasifikasi untuk mengelompokkan curah hujan ke dalam kelas: Tidak Hujan, Hujan Ringan, Hujan Sedang, Hujan Deras
2. Membangun Model Regresi untuk memprediksi nilai numerik dari curah hujan harian (dalam mm)
3. Mengembangkan strategi handling data tidak seimbang untuk kelas minoritas
4. Membersihkan dan menyiapkan data agar representatif dan tidak bias
---

## 📊 Data Understanding

### Dataset

Sumber: [Dataset Cuaca Harian dari Kaggle](https://www.kaggle.com/datasets/robbysaidiii/prediksi-cuaca-csv)

Dataset ini berisi data cuaca harian yang terdiri dari:

* Jumlah baris: 719 (pengamatan harian dari tahun 2022-2023)
* Jumlah kolom: 9

### Kondisi Data Awal

Sebelum dilakukan preprocessing, kondisi data adalah sebagai berikut:

* **Missing values ditemukan pada beberapa kolom:**

  * temp\_min: 2 nilai hilang
  * temp\_max: 5 nilai hilang
  * temp\_rata-rata: 3 nilai hilang
  * lembab\_rata-rata: 3 nilai hilang
  * cahaya\_jam: 4 nilai hilang
  * ch (target): 83 nilai hilang

* **Outlier dan nilai ekstrem:**

  * Nilai 9999 dan 8888 ditemukan pada beberapa kolom sebagai nilai ekstrim yang perlu diubah menjadi NAN .
  * Outlier alami juga diamati, lembab\_rata-rata 59, nilai ch 26, dan cahaya\_jam 38 yang berada di luar distribusi normal.

* **Visualisasi outlier:**
  ![visual\_outlier](https://github.com/Robbysaidiii/Machine_Learning_Terapan/blob/main/gambar/Cuplikan%20layar%202025-05-24%20231638.png)

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
 
 ![visual](https://github.com/Robbysaidiii/Machine_Learning_Terapan/blob/main/gambar/Cuplikan%20layar%202025-05-26%20233959.png)

## 🧹 Data Preparation

### Tahapan Persiapan Data

1. **Penanganan Nilai Ekstrem dan Missing Values**

   * Nilai 9999 dan 8888 diubah menjadi NaN agar dapat ditangani sebagai missing values.
   * Missing values diimputasi menggunakan nilai median setiap kolom, karena median lebih tahan terhadap outlier dibandingkan mean.
   * Setelah imputasi, dicek kembali dengan `.isnull().sum()` memastikan tidak ada data kosong tersisa.

2. **Konversi Format Tanggal**

   * Kolom Thn, bln, dan tgl digabung menjadi kolom datetime tunggal.
   * Kolom tanggal dijadikan index dataframe untuk menjaga urutan waktu.

3. **Kategorisasi Target (Label)**

   * ch == 0: Tidak Hujan
   * ch < 20: Hujan Ringan
   * ch < 50: Hujan Sedang
   * ch >= 50: Hujan Deras

4. **Encoding Label**

   * Label kategori curah hujan di-encode menggunakan LabelEncoder dari scikit-learn agar bisa digunakan model klasifikasi.

5. **Pemilihan Fitur**

   * Fitur yang digunakan: \['temp\_min', 'temp\_max', 'temp\_rata-rata', 'lembab\_rata-rata', 'cahaya\_jam']
   * Target: 'label' hasil encoding kategori curah hujan

6. **Normalisasi Data**

   * Fitur dinormalisasi ke rentang \[0, 1] menggunakan MinMaxScaler untuk menghindari fitur dengan skala besar mendominasi proses pelatihan model.

7. **Pembagian Dataset**

   * Dataset dibagi menjadi data latih dan uji dengan rasio 80%-20% menggunakan `train_test_split` tanpa shuffle agar urutan waktu tetap terjaga.

---

## 🤖 Modeling

### 1. Klasifikasi – Random Forest Classifier

**Penjelasan Algoritma:**
Random Forest adalah algoritma ensemble learning yang membangun banyak pohon keputusan (decision trees) pada subset data dan subset fitur yang berbeda. Hasil prediksi akhir diperoleh melalui voting mayoritas dari seluruh pohon. Pendekatan ini meningkatkan akurasi dan mengurangi risiko overfitting yang terjadi pada pohon tunggal.

**Keunggulan:**

* Robust terhadap data dengan noise dan fitur non-linear
* Dapat menangani data dengan distribusi yang kompleks
* Mampu mengestimasi pentingnya fitur

**Implementasi:**

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

**Parameter:**

* `n_estimators=100`: Jumlah pohon dalam hutan random forest
* `random_state=42`: Seed untuk reproduktibilitas

---

### 2. Regresi – Linear Regression

**Penjelasan Algoritma:**
Linear Regression memodelkan hubungan linier antara fitur input dan variabel target dengan mencari garis lurus yang paling sesuai, meminimalkan jumlah kuadrat kesalahan (residuals) antara prediksi dan nilai aktual.

**Implementasi:**

```python
model_reg = LinearRegression()
model_reg.fit(X_train, y_train_regresi)
```

---

## 🔎 Evaluation

### Hasil Evaluasi Klasifikasi (Random Forest):

| Kelas            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Tidak Hujan      | 0.88      | 0.88   | 0.88     | 96      |
| Hujan Ringan     | 0.65      | 0.76   | 0.70     | 42      |
| Hujan Sedang     | 0.00      | 0.00   | 0.00     | 6       |
| Hujan Deras      | -         | -      | -        | -       |
| **Accuracy**     |           |        | **0.81** | 144     |
| **Macro Avg**    | 0.51      | 0.55   | 0.53     | 144     |
| **Weighted Avg** | 0.78      | 0.81   | 0.79     | 144     |

**Interpretasi:**

* Akurasi keseluruhan mencapai 81%, cukup baik untuk klasifikasi dasar.
* Kinerja terbaik ada pada kelas mayoritas (Tidak Hujan).
* Model gagal mengklasifikasikan kelas minoritas “Hujan Sedang” akibat ketidakseimbangan data.

---

### Hasil Evaluasi Regresi (Linear Regression):

* Mean Squared Error (MSE): 0.6348
* R² Score: 0.2236

**Interpretasi:**

* R² Score hanya 0.22, menunjukkan model linear hanya mampu menjelaskan 22% variasi target.
* Model linier ini terlalu sederhana untuk menangkap kompleksitas pola curah hujan harian yang dipengaruhi banyak faktor.

---

### Perbandingan Model:

| Aspek      | Klasifikasi (Random Forest)      | Regresi (Linear Regression)         |
| ---------- | -------------------------------- | ----------------------------------- |
| Akurasi    | 81%                              | 22% (R² Score)                      |
| Kelebihan  | Robust, menangani non-linearitas | Cepat, sederhana                    |
| Kekurangan | Sensitif ketidakseimbangan kelas | Tidak mampu menangkap pola kompleks |

---

## 📌 Rekomendasi dan Kesimpulan

### Rekomendasi Perbaikan:

1. **Klasifikasi:**

   * Terapkan teknik penyeimbangan kelas seperti SMOTE, class weighting, atau undersampling.
   * Eksplorasi algoritma boosting (XGBoost, Gradient Boosting) yang sering kali lebih akurat.
   * Lakukan hyperparameter tuning menggunakan grid search atau random search untuk optimasi model.

2. **Regresi:**

   * Gunakan model regresi non-linear seperti Random Forest Regressor atau XGBoost Regressor.
   * Pertimbangkan transformasi target jika distribusi target sangat skewed.
   * Tambahkan teknik feature engineering agar fitur lebih representatif.

3. **Data:**

   * Sertakan visualisasi distribusi data sebelum dan sesudah preprocessing.
   * Perbaiki pengumpulan data agar mengurangi missing values dan outliers.

### Kesimpulan:

Model klasifikasi dengan Random Forest memberikan performa lebih baik dibanding regresi linear untuk prediksi curah hujan harian dengan akurasi 81%. Namun, ketidakseimbangan kelas menjadi kendala utama dalam klasifikasi beberapa kategori curah hujan minoritas seperti hujan sedang.

Model regresi linear kurang mampu menjelaskan variasi data secara signifikan, mengindikasikan perlunya model non-linear dan fitur yang lebih kaya.

Perbaikan pada teknik handling data dan eksplorasi model lebih canggih merupakan langkah penting untuk meningkatkan prediksi curah hujan harian yang lebih akurat dan bermanfaat untuk aplikasi nyata.
