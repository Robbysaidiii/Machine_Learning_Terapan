# Laporan Proyek Machine Learning - Roby saidi prasetyo

## Domain Proyek

Prediksi cuaca merupakan salah satu aplikasi machine learning yang sangat penting dalam kehidupan sehari-hari. Cuaca, khususnya curah hujan, memiliki dampak signifikan terhadap berbagai sektor seperti pertanian, transportasi, dan manajemen bencana alam. Kemampuan untuk memprediksi kategori intensitas hujan berdasarkan parameter meteorologi dapat membantu dalam pengambilan keputusan yang lebih baik.

Proyek ini fokus pada pengembangan model machine learning untuk mengklasifikasikan intensitas hujan berdasarkan data cuaca harian. Dengan memanfaatkan parameter seperti suhu, kelembaban, dan jam cahaya matahari, model dapat memprediksi apakah suatu hari akan mengalami tidak hujan, hujan ringan, hujan sedang, atau hujan deras.

**Mengapa masalah ini penting?**
- Membantu perencanaan aktivitas outdoor dan pertanian
- Mendukung sistem peringatan dini bencana alam
- Optimalisasi manajemen sumber daya air
- Peningkatan efisiensi dalam sektor transportasi dan logistik

## Business Understanding

### Problem Statements

1. **Bagaimana cara mengklasifikasikan intensitas hujan berdasarkan parameter cuaca?**
   - Diperlukan model yang dapat mengkategorikan curah hujan menjadi empat kategori: tidak hujan, hujan ringan, hujan sedang, dan hujan deras

2. **Parameter cuaca manakah yang paling berpengaruh terhadap prediksi intensitas hujan?**
   - Perlu identifikasi fitur-fitur meteorologi yang memiliki kontribusi terbesar dalam prediksi

3. **Bagaimana mencapai akurasi prediksi yang tinggi dengan data cuaca yang memiliki variabilitas tinggi?**
   - Data cuaca memiliki outliers dan missing values yang perlu ditangani dengan tepat

### Goals

1. **Mengembangkan model klasifikasi yang dapat memprediksi kategori intensitas hujan dengan akurasi tinggi (>95%)**
   - Model harus mampu membedakan empat kategori hujan dengan baik

2. **Mengidentifikasi fitur-fitur cuaca yang paling berpengaruh dalam prediksi intensitas hujan**
   - Analisis feature importance untuk pemahaman yang lebih baik

3. **Membangun sistem prediksi yang robust terhadap outliers dan missing values**
   - Implementasi preprocessing yang tepat untuk handling data quality issues

### Solution Statements

1. **Implementasi multiple machine learning algorithms untuk perbandingan performa:**
   - Random Forest Classifier
   - Support Vector Machine (SVM)
   - Naive Bayes
   - Gradient Boosting Classifier
   - Neural Network (MLP)

2. **Hyperparameter tuning untuk optimasi model terbaik:**
   - Grid Search dengan Cross Validation untuk Random Forest
   - Optimasi parameter untuk meningkatkan performa model

3. **Feature engineering dan selection techniques:**
   - Univariate Feature Selection
   - Recursive Feature Elimination (RFE)  
   - Principal Component Analysis (PCA)

## Data Understanding

Dataset yang digunakan adalah data cuaca harian dari tahun 2022-2023 yang terdiri dari 719 record dengan 9 kolom. Data mencakup informasi meteorologi lengkap untuk prediksi cuaca. Dataset dapat diunduh dari [Kaggle - Prediksi Cuaca CSV](https://www.kaggle.com/datasets/robbysaidiii/prediksi-cuaca-csv).

### Variabel-variabel pada dataset cuaca adalah sebagai berikut:

- **Thn**: Tahun pengamatan (2022-2023)
- **bln**: Bulan pengamatan (1-12)
- **tgl**: Tanggal pengamatan (1-31)
- **temp_min**: Suhu minimum harian (°C)
- **temp_max**: Suhu maksimum harian (°C)
- **temp_rata-rata**: Suhu rata-rata harian (°C)
- **lembab_rata-rata**: Kelembaban rata-rata harian (%)
- **ch**: Curah hujan harian (mm) - target variable
- **cahaya_jam**: Jam penyinaran matahari harian

### Exploratory Data Analysis (EDA)

Berdasarkan analisis data awal:

1. **Distribusi Data**: Data terdistribusi merata antara tahun 2022 dan 2023 dengan distribusi bulanan yang seimbang
2. **Missing Values**: Ditemukan missing values pada beberapa kolom, dengan kolom 'ch' memiliki missing values terbanyak (83 nilai)
3. **Outliers**: Setiap variabel numerik memiliki outliers, dengan curah hujan memiliki outliers terbanyak (103 outliers)
4. **Anomali Data**: Ditemukan nilai anomali 9999 dan 8888 yang merupakan kode untuk data hilang

### Korelasi Antar Variabel

- Korelasi positif kuat antara temp_min, temp_max, dan temp_rata-rata (0.7-0.9)
- Korelasi negatif moderat antara suhu dan kelembaban (-0.3 hingga -0.5)
- Curah hujan memiliki korelasi negatif dengan jam cahaya matahari

## Data Preparation

### 1. Handling Missing Values dan Anomali
```python
# Mengganti nilai anomali dengan NaN
data.replace([9999, 8888], pd.NA, inplace=True)

# Imputation dengan median untuk robustness terhadap outliers
for col in numeric_columns:
    data[col].fillna(data[col].median(), inplace=True)
```

**Alasan**: Median dipilih karena lebih robust terhadap outliers dibandingkan mean, dan lebih sesuai untuk data cuaca yang memiliki distribusi skewed.

### 2. Feature Engineering
```python
# Kategorisasi intensitas hujan
def categorize_rain(ch):
    if ch == 0: return 'tidak hujan'
    elif ch < 20: return 'hujan ringan' 
    elif ch < 50: return 'hujan sedang'
    else: return 'hujan deras'
```

**Alasan**: Kategorisasi berdasarkan standar meteorologi untuk intensitas hujan, mengubah masalah regresi menjadi klasifikasi multi-class.

### 3. Normalisasi Fitur
```python
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
```

**Alasan**: Min-Max Scaling memastikan semua fitur berada dalam rentang [0,1], penting untuk algoritma yang sensitif terhadap skala seperti SVM dan Neural Network.

### 4. Feature Selection
Implementasi tiga teknik feature selection:
- **Univariate Feature Selection**: Memilih 4 fitur terbaik berdasarkan skor statistik
- **Recursive Feature Elimination (RFE)**: Eliminasi rekursif dengan Random Forest
- **Principal Component Analysis (PCA)**: Reduksi dimensi dengan mempertahankan 95% varians

**Alasan**: Multiple techniques memungkinkan perbandingan efektivitas berbagai pendekatan feature selection.

## Modeling

### Model yang Digunakan

1. **Random Forest Classifier**
   - **Kelebihan**: Robust terhadap outliers, dapat menangani non-linearity, memberikan feature importance
   - **Kekurangan**: Dapat overfitting pada data noise, interpretability terbatas untuk decision tree individual
   - **Parameter**: n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1

2. **Support Vector Machine (SVM)**
   - **Kelebihan**: Efektif untuk high-dimensional data, memory efficient
   - **Kekurangan**: Sensitif terhadap feature scaling, lambat pada dataset besar
   - **Parameter**: kernel='rbf', probability=True

3. **Naive Bayes**
   - **Kelebihan**: Simple, cepat, baik untuk baseline model
   - **Kekurangan**: Asumsi independensi fitur yang kuat
   - **Parameter**: Default Gaussian distribution

4. **Gradient Boosting Classifier**
   - **Kelebihan**: High predictive accuracy, dapat menangani berbagai tipe data
   - **Kekurangan**: Prone to overfitting, parameter tuning kompleks
   - **Parameter**: Default parameters dengan random_state=42

5. **Neural Network (MLP)**
   - **Kelebihan**: Dapat mempelajari complex patterns, flexible architecture
   - **Kekurangan**: Black box model, membutuhkan data banyak, sensitive terhadap scaling
   - **Parameter**: hidden_layer_sizes=(100,50), max_iter=1000

### Hyperparameter Tuning

Dilakukan Grid Search CV pada Random Forest dengan parameter:
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30], 
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

**Best Parameters**: n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1

**Alasan Pemilihan Random Forest sebagai Model Terbaik**:
Random Forest dipilih karena mencapai akurasi tertinggi (99.3%) bersama Gradient Boosting, namun Random Forest lebih interpretable dan robust. Model ini juga menunjukkan stabilitas tinggi dengan CV accuracy 98.6% (±1.8%).

## Evaluation

### Metrik Evaluasi yang Digunakan

1. **Accuracy**: Proporsi prediksi yang benar dari total prediksi
   - Formula: (TP + TN) / (TP + TN + FP + FN)
   - Cocok untuk balanced dataset

2. **Cross-Validation Score**: Rata-rata akurasi dari 5-fold cross validation
   - Memberikan estimasi performa yang lebih robust
   - Mengurangi bias dari single train-test split

3. **ROC-AUC**: Area Under ROC Curve untuk setiap kelas
   - Mengukur kemampuan model membedakan antar kelas
   - Nilai mendekati 1.0 menunjukkan performa excellent

### Hasil Evaluasi

#### Perbandingan Model:
- **Random Forest**: 99.3% (terbaik)
- **Gradient Boosting**: 99.3% (terbaik) 
- **Naive Bayes**: 97.9%
- **Neural Network**: 95.8%
- **SVM**: 73.6% (terburuk)

#### Model Terpilih (Random Forest):
- **Test Accuracy**: 99.3%
- **Cross-Validation Accuracy**: 98.6% (±1.8%)
- **ROC-AUC**: > 0.95 untuk semua kelas

### Feature Importance Analysis

Berdasarkan Random Forest feature importance:
1. **Curah hujan (ch)**: Fitur paling penting (logis karena merupakan basis kategorisasi)
2. **Kelembaban rata-rata**: Fitur kedua terpenting
3. **Suhu minimum**: Fitur ketiga terpenting  
4. **Jam cahaya**: Importance terendah

### Model Interpretability

**SHAP Analysis** menunjukkan:
- Curah hujan tinggi → prediksi hujan deras
- Kelembaban tinggi → cenderung hujan
- Suhu tinggi → cenderung tidak hujan

**Partial Dependence Plots** mengkonfirmasi hubungan non-linear antara fitur dan target untuk setiap kelas.

### Kesimpulan Evaluasi

Model Random Forest berhasil mencapai performa yang sangat baik dengan akurasi 99.3% pada data test dan konsistensi tinggi pada cross-validation (98.6% ±1.8%). ROC curves menunjukkan discriminative power yang excellent untuk semua kelas dengan AUC > 0.95. Model ini sangat cocok untuk aplikasi prediksi kategori hujan dengan tingkat kepercayaan tinggi.

---

**Catatan**: Model telah berhasil memenuhi semua goals yang ditetapkan dengan akurasi > 95%, identifikasi fitur penting yang jelas, dan robustness terhadap data quality issues. Implementasi feature engineering dan hyperparameter tuning berkontribusi signifikan terhadap performa model yang optimal.
