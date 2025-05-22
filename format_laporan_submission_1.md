
# 📊 Laporan Proyek Machine Learning - Robysaidi

**Judul:** Prediksi Kategori Hujan Harian Berdasarkan Data Cuaca Menggunakan Random Forest

---

## 📘 Domain Proyek

Permasalahan iklim dan prediksi cuaca semakin penting di era perubahan iklim. Salah satu hal krusial adalah memprediksi **kategori curah hujan** secara otomatis dari data cuaca harian. Prediksi ini sangat berguna dalam sektor pertanian, logistik, hingga mitigasi bencana.

Beberapa penelitian terdahulu telah menunjukkan bahwa algoritma machine learning seperti Random Forest dan KNN dapat digunakan secara efektif untuk klasifikasi cuaca. Dengan menggunakan data historis yang mencakup suhu, kelembaban, dan lama penyinaran matahari, sistem prediksi dapat dikembangkan untuk mendukung keputusan berbasis cuaca.

Referensi:

* Nuruzzaman et al., “Rainfall Prediction using Machine Learning,” *Journal of Applied Sciences*, 2021.
* BMKG (Badan Meteorologi, Klimatologi, dan Geofisika) Data Repository: [bmkg.go.id](https://www.bmkg.go.id/)

---

## 🎯 Business Understanding

### Problem Statements

1. Bagaimana cara mengklasifikasikan curah hujan harian ke dalam kategori: tidak hujan, hujan ringan, sedang, dan deras?
2. Fitur cuaca apa yang paling relevan dalam menentukan kategori hujan?

### Goals

1. Membangun model machine learning untuk memprediksi kategori hujan menggunakan data suhu, kelembaban, dan penyinaran matahari.
2. Mengevaluasi performa model dengan metrik klasifikasi seperti akurasi dan f1-score.

### Solution Statements

* Model klasifikasi Random Forest digunakan karena kemampuannya mengatasi data tabular dan ketidakseimbangan kelas.
* Proses preprocessing mencakup normalisasi fitur, encoding label, dan analisis outlier.
* Model akan dievaluasi menggunakan confusion matrix dan metrik evaluasi lainnya.

---

## 🧠 Data Understanding

Dataset berisi 719 data harian dari tahun 2022 hingga akhir 2023. Data diperoleh dari file `data_cuaca.csv` yang disimpan di Google Drive. Dataset terdiri dari kolom:

* `Thn`, `bln`, `tgl` : tanggal pencatatan
* `temp_min`, `temp_max`, `temp_rata-rata` : suhu harian
* `lembab_rata-rata` : kelembaban rata-rata
* `ch` : curah hujan dalam mm (target asli)
* `cahaya_jam` : lama penyinaran matahari per hari

Visualisasi distribusi fitur dan boxplot menunjukkan adanya outlier, terutama pada fitur `ch` (curah hujan).

---

## 🧹 Data Preparation

1. **Cleaning Data:**

   * Nilai ekstrem seperti 9999 dan 8888 diubah menjadi NaN dan diisi dengan median.
   * Duplikat data dicek dan tidak ditemukan.

2. **Feature Engineering:**

   * Kolom tanggal digabung dan dijadikan index.
   * Fitur `ch` dikategorikan ke dalam 4 kelas:

     * `tidak hujan` (ch = 0)
     * `hujan ringan` (ch < 20 mm)
     * `hujan sedang` (20 ≤ ch < 50)
     * `hujan deras` (ch ≥ 50)

3. **Encoding dan Scaling:**

   * LabelEncoder digunakan untuk mengubah kategori hujan menjadi angka 0-3.
   * MinMaxScaler digunakan untuk normalisasi fitur numerik (`temp_min` hingga `cahaya_jam`).

4. **Splitting Data:**

   * Data dibagi menjadi 80% training dan 20% testing tanpa shuffle karena data bersifat time series.

---

## 🤖 Modeling

Model yang digunakan: **Random Forest Classifier**

* Alasan: cocok untuk data tabular dan robust terhadap outlier serta kelas tidak seimbang.
* Parameter: `n_estimators=100`, `random_state=42`
* Model disimpan dalam file `.pkl` untuk digunakan kembali (inference).

---

## 📈 Evaluation

### Metrik Evaluasi:

* **Accuracy**: 80.00%
* **Precision (weighted)**: 0.77
* **Recall (weighted)**: 0.80
* **F1-score (weighted)**: 0.78

### Classification Report:

```
                precision    recall  f1-score   support

tidak hujan       0.88      0.88      0.88        96
hujan ringan      0.65      0.74      0.69        42
hujan sedang      0.00      0.00      0.00         6
```

### Confusion Matrix:

Model masih kesulitan mengenali kelas “hujan sedang” karena datanya sedikit (hanya 6).

![Confusion Matrix](visual_confusion_matrix.png)

---

### Kesimpulan:

* Model Random Forest memberikan akurasi tinggi untuk dua kelas utama (`tidak hujan` dan `hujan ringan`).
* Perlu balancing data atau teknik oversampling untuk memperbaiki prediksi pada kelas minoritas seperti `hujan sedang`.
* Proyek dapat ditingkatkan dengan teknik tuning (GridSearch) dan perbandingan model lain seperti XGBoost atau SVM.

---

### 📦 File Submission:

* `robbysaidi_machine_learning_terapan.ipynb` (Notebook dengan penjelasan dan kode)
* `robbysaidi_machine_learning_terapan.py` (Model Random Forest terlatih)


---

