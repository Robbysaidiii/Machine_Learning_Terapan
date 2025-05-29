Berikut versi laporan proyek machine learning yang sudah rapih dan tanpa ada kata "di rumah" atau semacamnya, menggunakan format Markdown sesuai permintaanmu:

````markdown
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
3. Membersihkan dan menyiapkan data agar representatif dan tidak bias

## 📊 Data Understanding

### Dataset
Sumber: [Dataset Cuaca Harian](https://example.com/link-dataset) (tautan contoh, ganti dengan link dataset sebenarnya)

Dataset ini berisi data cuaca harian yang terdiri dari:
- Jumlah baris: 719 (pengamatan harian dari tahun 2022-2023)
- Jumlah kolom: 9

### Kondisi Data Awal
- Missing values:
  - temp_min: 2 nilai hilang
  - temp_max: 5 nilai hilang  
  - temp_rata-rata: 3 nilai hilang
  - lembab_rata-rata: 3 nilai hilang
  - cahaya_jam: 4 nilai hilang
  - ch: 83 nilai hilang
- Outliers:
  - Nilai ekstrem 9999 dan 8888 ditemukan pada beberapa kolom
  - Outlier alami ditemukan pada kolom lembab_rata-rata (59), ch (26), dan cahaya_jam (38)

### Variabel
| Kolom       | Deskripsi                         |
|-------------|---------------------------------|
| Thn, bln, tgl | Tanggal pencatatan             |
| temp_min    | Suhu minimum harian (°C)         |
| temp_max    | Suhu maksimum harian (°C)        |  
| temp_rata-rata | Suhu rata-rata harian (°C)     |
| lembab_rata-rata | Kelembaban rata-rata harian (%) |
| ch          | Curah hujan (mm) – target        |
| cahaya_jam  | Lama penyinaran matahari (jam)   |

## 🧹 Data Preparation

### Tahapan Persiapan Data:
1. **Penanganan Nilai Ekstrem**:
   - Mengganti nilai 9999 dan 8888 dengan NaN
   - Imputasi nilai hilang dengan median (robust terhadap outlier)

2. **Konversi Format Tanggal**:
   - Menggabungkan kolom Thn, bln, tgl menjadi satu kolom datetime
   - Menjadikan kolom tanggal sebagai index

3. **Kategorisasi Target**:
   - ch == 0: tidak hujan  
   - ch < 20: hujan ringan
   - ch < 50: hujan sedang
   - ch >= 50: hujan deras

4. **Encoding Label**:
   - Menggunakan LabelEncoder untuk mengubah kategori menjadi nilai numerik

5. **Pemilihan Fitur**:
   - Fitur yang digunakan: ['temp_min', 'temp_max', 'temp_rata-rata', 'lembab_rata-rata', 'cahaya_jam']
   - Target: 'label' (hasil encoding kategori hujan)

6. **Normalisasi Data**:
   - Menggunakan MinMaxScaler untuk menormalkan fitur ke rentang [0,1]

7. **Pembagian Data**:
   - Train-test split 80%-20% tanpa shuffle untuk menjaga urutan waktu

## 🤖 Modeling

### 1. Klasifikasi – Random Forest Classifier

**Penjelasan Algoritma**:
Random Forest adalah metode ensemble learning yang membangun banyak pohon keputusan dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting. Setiap pohon dilatih pada subset data dan fitur yang berbeda, kemudian hasil prediksi ditentukan melalui voting mayoritas.

**Implementasi**:
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
````

**Parameter**:

* n\_estimators=100: Jumlah pohon dalam forest
* random\_state=42: Untuk reproduktibilitas hasil

### 2. Regresi – Linear Regression

**Penjelasan Algoritma**:
Linear Regression memodelkan hubungan linear antara variabel independen (fitur) dan dependen (target) dengan mencari garis lurus yang paling sesuai dengan data.

**Implementasi**:

```python
model_reg = LinearRegression()
model_reg.fit(X_train, y_train)
```

## 🔎 Evaluation

### Hasil Evaluasi Klasifikasi (Random Forest):

```
              precision    recall  f1-score   support

hujan ringan       0.65      0.76      0.70        42
hujan sedang       0.00      0.00      0.00         6
 tidak hujan       0.88      0.88      0.88        96

    accuracy                           0.81       144
   macro avg       0.51      0.55      0.53       144
weighted avg       0.78      0.81      0.79       144
```

**Interpretasi**:

* Akurasi keseluruhan: 81% - baik untuk klasifikasi dasar
* Performa bagus untuk kelas dominan (tidak hujan)
* Gagal mengklasifikasikan "hujan sedang" karena data sangat sedikit (hanya 6 sampel)

### Hasil Evaluasi Regresi (Linear Regression):

```
Mean Squared Error: 0.6348
R² Score: 0.2236
```

**Interpretasi**:

* R² Score = 0.22 → hanya menjelaskan 22% variasi dalam target
* Model linier terlalu sederhana untuk fenomena iklim yang kompleks

### Perbandingan Model:

| Aspek      | Klasifikasi (RF)                  | Regresi (Linear)                   |
| ---------- | --------------------------------- | ---------------------------------- |
| Akurasi    | 81%                               | 22% (setelah pembulatan)           |
| Kelebihan  | Robust, cocok untuk data kategori | Sederhana, cepat                   |
| Kekurangan | Sensitif terhadap imbalance kelas | Tidak bisa menangkap pola kompleks |

## 📌 Rekomendasi dan Kesimpulan

### Rekomendasi Perbaikan:

1. Untuk Klasifikasi:

   * Gunakan teknik handling imbalance class (SMOTE atau class weights)
   * Coba model boosting seperti XGBoost atau Gradient Boosting
   * Lakukan hyperparameter tuning

2. Untuk Regresi:

   * Gunakan model non-linear seperti Random Forest Regressor
   * Pertimbangkan transformasi target jika distribusi skewed
   * Tambahkan feature engineering

### Kesimpulan:

Pendekatan klasifikasi dengan Random Forest memberikan hasil yang lebih baik (akurasi 81%) dibanding regresi linier untuk memprediksi curah hujan. Namun, model masih kesulitan memprediksi kelas minoritas (hujan sedang) karena ketidakseimbangan data.

Langkah selanjutnya adalah memperbaiki model dengan teknik handling imbalance class dan mencoba algoritma yang lebih advanced untuk meningkatkan performa prediksi.

```

Kalau ada yang mau kamu tambah atau ubah, bilang aja ya!
```
