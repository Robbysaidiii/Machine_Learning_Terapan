---

#  Laporan Proyek Machine Learning – Roby saidi prasetyo

##  Domain Proyek

Perubahan iklim telah membuat pola cuaca menjadi semakin sulit diprediksi, terutama dalam hal curah hujan. Hal ini berdampak besar pada sektor pertanian, transportasi, konstruksi, hingga mitigasi bencana alam. Oleh karena itu, pengembangan model prediksi curah hujan yang akurat menjadi kebutuhan mendesak.

Model machine learning memberikan pendekatan fleksibel dan efisien dibanding metode statistik konvensional karena kemampuannya menangkap pola non-linear dan kompleks.

###  Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan?

Prediksi curah hujan dapat membantu:

* **Petani** dalam menentukan waktu tanam dan panen
* **Pemerintah** dalam merancang sistem peringatan dini
* **Kontraktor dan perencana** dalam proyek pembangunan infrastruktur

###  Referensi Ilmiah

1. Pradhan et al., “Rainfall prediction using machine learning classification techniques,” *Materials Today: Proceedings*, vol. 62, 2022.
2. Hussain et al., “Comparative Analysis of Machine Learning Algorithms for Rainfall Prediction,” *IJERT*, 2021.

---

##  Business Understanding

###  Problem Statements

1. Bagaimana memprediksi tingkat curah hujan berdasarkan fitur cuaca seperti suhu, kelembaban, dan durasi penyinaran matahari?
2. Bagaimana mengelola data yang mengandung nilai ekstrem (9999, 8888) dan nilai hilang?
3. Model seperti apa yang lebih efektif: klasifikasi kategori curah hujan, atau prediksi nilai curah hujan aktual?

###  Goals

1. Membangun Model Klasifikasi untuk mengelompokkan curah hujan ke dalam kelas: Tidak Hujan, Hujan Ringan, Hujan Sedang, Hujan Deras
2. Membangun Model Regresi untuk memprediksi nilai numerik dari curah hujan harian (dalam mm)
3. Mengembangkan strategi handling data tidak seimbang untuk kelas minoritas
4. Membersihkan dan menyiapkan data agar representatif dan tidak bias

---

##  Data Understanding

### Dataset

* Sumber: [Kaggle – Prediksi Cuaca CSV](https://www.kaggle.com/datasets/robbysaidiii/prediksi-cuaca-csv)
* Jumlah baris: 719 (pengamatan harian dari tahun 2022–2023)
* Jumlah kolom: 9

### Variabel

| Kolom             | Deskripsi                       |
| ----------------- | ------------------------------- |
| Thn, bln, tgl     | Tanggal pencatatan              |
| temp\_min         | Suhu minimum harian (°C)        |
| temp\_max         | Suhu maksimum harian (°C)       |
| temp\_rata-rata   | Suhu rata-rata harian (°C)      |
| lembab\_rata-rata | Kelembaban rata-rata harian (%) |
| cahaya\_jam       | Lama penyinaran matahari (jam)  |
| ch                | Curah hujan (mm) – **Target**   |

### Kondisi Data Awal

* **Missing values ditemukan:**

  * `temp_min`: 2
  * `temp_max`: 5
  * `temp_rata-rata`: 3
  * `lembab_rata-rata`: 3
  * `cahaya_jam`: 4
  * `ch`: 83

* **Nilai Ekstrem (outliers):**

  * Nilai 9999 dan 8888 ditemukan → diubah menjadi `NaN`
  * Nilai outlier alami: `lembab_rata-rata = 59`, `ch = 26`, `cahaya_jam = 38`

![Outlier Visualization](https://github.com/Robbysaidiii/Machine_Learning_Terapan/blob/main/gambar/Cuplikan%20layar%202025-05-24%20231638.png)

---

##  Data Preparation

### Tahapan:

1. **Ubah nilai ekstrem ke `NaN`** → memudahkan proses imputasi
2. **Imputasi missing values** dengan **median** → tahan terhadap outlier
3. **Gabungkan kolom tanggal** menjadi satu kolom `datetime`, dijadikan index
4. **Kategorisasi target `ch`:**

   * `0 mm`: Tidak Hujan
   * `<20 mm`: Hujan Ringan
   * `<50 mm`: Hujan Sedang
   * `>=50 mm`: Hujan Deras
5. **Label Encoding** target → agar bisa digunakan model klasifikasi
6. **Normalisasi fitur** menggunakan `MinMaxScaler` ke rentang \[0, 1]
7. **Split dataset**: 80% data latih – 20% data uji (tanpa shuffle)

---

##  Modeling

###  Model 1: Random Forest Classifier

**Penjelasan:**
Random Forest membangun banyak decision tree dan menggabungkan hasil prediksi mereka. Sangat cocok untuk klasifikasi non-linear dan dataset dengan outlier.

**Implementasi:**

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

### 📐 Model 2: Linear Regression

**Penjelasan:**
Model regresi linier memprediksi nilai `ch` berdasarkan hubungan linier antar fitur. Cocok sebagai baseline sederhana.

**Implementasi:**

```python
from sklearn.linear_model import LinearRegression
model_reg = LinearRegression()
model_reg.fit(X_train, y_train_regresi)
```

---

## 🔎 Evaluation

### 📈 Evaluasi Klasifikasi – Random Forest

| Kelas            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Tidak Hujan      | 0.88      | 0.88   | 0.88     | 96      |
| Hujan Ringan     | 0.65      | 0.76   | 0.70     | 42      |
| Hujan Sedang     | 0.00      | 0.00   | 0.00     | 6       |
| Hujan Deras      | -         | -      | -        | -       |
| **Accuracy**     |           |        | **0.81** | 144     |
| **Macro Avg**    | 0.51      | 0.55   | 0.53     | 144     |
| **Weighted Avg** | 0.78      | 0.81   | 0.79     | 144     |

>  Model gagal mengenali kelas minoritas karena ketidakseimbangan kelas.

---

### 📉 Evaluasi Regresi – Linear Regression

* **Mean Squared Error (MSE):** 0.6348
* **R² Score:** 0.2236

> R² Score hanya 0.22 → model linier terlalu sederhana untuk menangkap pola kompleks pada curah hujan.

---

###  Perbandingan Model

| Aspek      | Klasifikasi (Random Forest)       | Regresi (Linear Regression)            |
| ---------- | --------------------------------- | -------------------------------------- |
| Akurasi    | 81%                               | R²: 22%                                |
| Kelebihan  | Menangani non-linearitas          | Cepat, mudah                           |
| Kekurangan | Sensitif terhadap kelas minoritas | Tidak fleksibel terhadap pola kompleks |

---

##  Rekomendasi & Kesimpulan

###  Rekomendasi

**Untuk Klasifikasi:**

* Gunakan teknik balancing seperti **SMOTE**, **undersampling**, atau **class weights**
* Coba algoritma lain: **XGBoost**, **Gradient Boosting**
* Lakukan **hyperparameter tuning** dengan GridSearch

**Untuk Regresi:**

* Gunakan model non-linear seperti **Random Forest Regressor**, **SVR**, atau **XGBoost Regressor**
* Lakukan transformasi pada target jika distribusinya tidak normal
* Tambahkan fitur baru seperti histori cuaca, tekanan udara, dll

**Untuk Data:**

* Lakukan eksplorasi visual sebelum dan sesudah preprocessing
* Tingkatkan kualitas dan kuantitas data untuk memperbaiki generalisasi model

---

###  Kesimpulan

Model klasifikasi dengan Random Forest menunjukkan kinerja cukup baik dengan akurasi 81%, namun masih lemah dalam mengenali kelas minoritas. Sebaliknya, model regresi linier kurang cocok untuk memodelkan curah hujan harian karena hanya menjelaskan 22% variabilitas.

Prediksi curah hujan harian lebih efektif dilakukan dengan pendekatan **klasifikasi multikelas** menggunakan model non-linear yang dilatih pada data bersih dan seimbang.

---
