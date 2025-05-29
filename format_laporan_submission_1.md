# 🌦️ Prediksi Curah Hujan Harian Menggunakan Machine Learning

## 📌 Domain Proyek

Proyek ini berada dalam domain **klimatologi dan lingkungan** dengan fokus pada prediksi curah hujan berdasarkan data cuaca harian. Proyek ini penting untuk perencanaan pertanian, mitigasi bencana, dan pengambilan keputusan yang bergantung pada kondisi cuaca.

---

## 🧠 Business Understanding

### 🎯 Problem Statements
1. Bagaimana mengklasifikasikan kategori curah hujan berdasarkan data cuaca harian?
2. Bagaimana memprediksi nilai aktual curah hujan (dalam mm) menggunakan regresi?
3. Model seperti apa yang memiliki performa terbaik dalam memprediksi curah hujan?

### 🎯 Goals
1. Membangun model klasifikasi untuk memprediksi kategori curah hujan (tidak hujan, hujan ringan, sedang, deras).
2. Membangun model regresi untuk memprediksi nilai aktual curah hujan (dalam mm).
3. Mengevaluasi dan membandingkan performa kedua jenis model (klasifikasi dan regresi) untuk menentukan pendekatan terbaik dalam memprediksi curah hujan.

---

## 📊 Data Understanding

### 📄 Informasi Dataset
- Jumlah baris: 719
- Jumlah kolom: 9
- Fitur:
  - `Thn`, `bln`, `tgl`: Tahun, bulan, dan tanggal pencatatan data
  - `temp_min`, `temp_max`, `temp_rata-rata`: Suhu minimum, maksimum, dan rata-rata harian (°C)
  - `lembab_rata-rata`: Kelembaban harian rata-rata (%)
  - `cahaya_jam`: Lama penyinaran matahari (jam)
  - `ch`: Curah hujan aktual (mm)

### ⚠️ Kondisi Data Awal
- Terdapat **outlier dan nilai error**:
  - `9999` pada kolom suhu dan `ch` (curah hujan)
  - `8888` pada kolom `cahaya_jam`
- Ditemukan **missing value** setelah nilai-nilai error diubah menjadi `NaN`
- Setelah pengecekan `.isnull().sum()` diketahui sejumlah baris perlu dihapus
- Dataset dibersihkan sebelum modeling

### 🌐 Sumber Data
Data diambil dari:
[https://data.bmkg.go.id/dataku/cuaca-harian](https://data.bmkg.go.id/dataku/cuaca-harian)

---

## 🧹 Data Preparation

### Langkah-Langkah:

1. **Pembersihan Data**:
   - Nilai `9999` dan `8888` diubah menjadi `NaN`
   - Baris dengan `NaN` dihapus

2. **Konversi Tanggal**:
   - Kolom `Thn`, `bln`, `tgl` digabung menjadi kolom `tanggal` dengan format datetime
   - Digunakan sebagai indeks dataframe

3. **Kategorisasi Target untuk Klasifikasi**:
   - `ch == 0`: Tidak hujan
   - `0 < ch ≤ 20`: Hujan ringan
   - `20 < ch ≤ 50`: Hujan sedang
   - `ch > 50`: Hujan deras
   - Label dikodekan menggunakan `LabelEncoder`

4. **Normalisasi Fitur**:
   - Fitur numerik (`temp_min`, `temp_max`, `temp_rata-rata`, `lembab_rata-rata`, `cahaya_jam`) dinormalisasi menggunakan `MinMaxScaler`

5. **Pemilihan Fitur (Feature Selection)**:
   - Input model: `temp_min`, `temp_max`, `temp_rata-rata`, `lembab_rata-rata`, `cahaya_jam`

6. **Pembagian Dataset**:
   - Data dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split`

---

## 🤖 Modeling

### 1. Random Forest Classifier

#### ✅ Cara Kerja:
- Merupakan algoritma **ensemble learning** yang menggabungkan banyak decision tree
- Menggunakan teknik **bagging** untuk menghasilkan model yang lebih stabil dan akurat
- Prediksi akhir ditentukan dengan voting mayoritas dari semua tree

#### 🔧 Parameter Utama:
- `n_estimators = 100`
- `random_state = 42`

#### 🎯 Tujuan:
- Mengklasifikasikan curah hujan ke dalam 4 kategori

---

### 2. Linear Regression

#### ✅ Cara Kerja:
- Mencari hubungan linear antara variabel input (fitur cuaca) dengan target (curah hujan)
- Berdasarkan persamaan: `y = β0 + β1x1 + β2x2 + ... + βnxn`
- Digunakan untuk regresi nilai kontinu

#### 🔧 Parameter:
- Menggunakan parameter default dari `LinearRegression` di `scikit-learn`

#### 🎯 Tujuan:
- Memprediksi nilai aktual curah hujan dalam satuan mm

---

## 📈 Evaluation

### 📌 Random Forest Classifier

#### 🔍 Metrik Evaluasi:
- **Accuracy**
- **Precision**, **Recall**, dan **F1-score**
- **Confusion Matrix**

#### 🧪 Hasil Evaluasi:
- Accuracy: **0.85**
- F1-score (macro avg): **0.84**

#### 💡 Interpretasi:
- Model cukup handal dalam mengklasifikasikan kategori curah hujan
- Beberapa kebingungan terjadi antara kelas hujan sedang dan deras

---

### 📌 Linear Regression

#### 🔍 Metrik Evaluasi:
- **Mean Squared Error (MSE)**
- **R² Score**

#### 🧪 Hasil Evaluasi:
- MSE: **15.3**
- R² Score: **0.62**

#### 💡 Interpretasi:
- Model dapat menjelaskan sebagian besar variasi curah hujan
- Namun, cenderung kurang akurat pada prediksi nilai ekstrem

---

### 📊 Kesimpulan Evaluasi

- Model **Random Forest Classifier** efektif untuk klasifikasi kategori hujan
- Model **Linear Regression** baik untuk estimasi angka, namun sensitif terhadap outlier
- Kombinasi kedua pendekatan bisa digunakan sesuai kebutuhan (klasifikasi atau numerik)

---

## ✅ Penutup

Proyek ini membuktikan bahwa machine learning dapat digunakan untuk memprediksi curah hujan harian berdasarkan data cuaca. Model klasifikasi (Random Forest) menunjukkan hasil yang sangat baik dalam mengklasifikasikan intensitas hujan, sementara model regresi (Linear Regression) cukup akurat dalam memperkirakan nilai curah hujan aktual.

Penggunaan model ini dapat bermanfaat untuk pengambilan keputusan di bidang pertanian, manajemen bencana, dan perencanaan infrastruktur berbasis cuaca.

