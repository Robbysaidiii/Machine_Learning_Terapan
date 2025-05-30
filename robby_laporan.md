 Laporan Proyek Machine Learning - Robysaidi

## 1. Domain Proyek

### Latar Belakang

Industri perfilman merupakan salah satu sektor ekonomi kreatif terbesar di dunia dengan nilai pasar global mencapai lebih dari $100 miliar USD per tahun. Namun, industri ini juga dikenal memiliki tingkat risiko yang tinggi karena tidak semua film yang diproduksi dapat mencapai kesuksesan komersial. Menurut data dari Motion Picture Association (MPA), hanya sekitar 20-30% film yang dirilis setiap tahunnya yang dapat mencapai break-even point atau menghasilkan keuntungan.

Prediksi kesuksesan film menjadi sangat penting bagi berbagai stakeholder dalam industri perfilman, termasuk:
- **Produser dan Studio**: Untuk mengalokasikan anggaran produksi dan pemasaran secara efektif
- **Investor**: Untuk menilai potensi return on investment (ROI) sebelum membiayai proyek film
- **Distributor**: Untuk merencanakan strategi distribusi dan jadwal rilis yang optimal
- **Eksibitor**: Untuk mengatur alokasi layar bioskop berdasarkan prediksi demand

Dengan kemajuan teknologi machine learning, kini dimungkinkan untuk mengembangkan model prediktif yang dapat membantu mengidentifikasi faktor-faktor yang berkontribusi terhadap kesuksesan komersial sebuah film berdasarkan data historis.

### Mengapa Masalah Ini Harus Diselesaikan

1. **Mitigasi Risiko Finansial**: Industri film memiliki tingkat kegagalan yang sangat tinggi, dengan mayoritas film tidak mencapai profitabilitas
2. **Optimalisasi Alokasi Sumber Daya**: Dengan prediksi yang akurat, sumber daya dapat dialokasikan lebih efisien
3. **Pengambilan Keputusan Berbasis Data**: Mengurangi ketergantungan pada intuisi dan "gut feeling" dalam industri yang sangat kompetitif
4. **Peningkatan ROI**: Membantu investor dan studio membuat keputusan investasi yang lebih informed

### Referensi

- Motion Picture Association. (2023). "Theme Report: Theatrical and Home Entertainment Market Environment (THEME) 2022". [MPA Official Website](https://www.motionpictures.org/)
- Follows, S. (2023). "Film Data and Education". [Stephen Follows Research](https://stephenfollows.com/)
- Panaligan, R., & Chen, A. (2013). "Quantifying Movie Magic with Google Search". Google Research Blog.
- McKenzie, J. (2023). "Box Office Prediction Using Machine Learning: A Comprehensive Analysis". Journal of Entertainment Technology, 15(2), 45-67.

## 2. Business Understanding

### Problem Statements

Berdasarkan analisis domain proyek, dapat diidentifikasi beberapa pernyataan masalah sebagai berikut:

1. **Bagaimana cara memprediksi kesuksesan komersial sebuah film berdasarkan karakteristik film tersebut sebelum dirilis?**
   - Tantangan utama adalah mengidentifikasi fitur-fitur yang paling berpengaruh terhadap performa box office
   - Diperlukan definisi yang jelas mengenai "kesuksesan" dalam konteks komersial

2. **Faktor-faktor apa saja yang paling berpengaruh terhadap kesuksesan komersial sebuah film?**
   - Perlu dilakukan analisis feature importance untuk memahami kontribusi setiap variabel
   - Insight ini dapat membantu stakeholder fokus pada aspek-aspek yang paling penting

3. **Seberapa akurat prediksi yang dapat dicapai dengan menggunakan pendekatan machine learning?**
   - Perlu evaluasi komprehensif untuk memastikan model dapat diandalkan dalam praktik bisnis
   - Diperlukan perbandingan multiple algorithms untuk mendapat solusi optimal

### Goals

Tujuan utama dari proyek ini adalah:

1. **Membangun model prediktif yang akurat** untuk mengklasifikasikan kesuksesan komersial film dengan metrik evaluasi yang sesuai konteks bisnis
2. **Mengidentifikasi faktor-faktor kunci** yang paling berpengaruh terhadap kesuksesan film melalui analisis feature importance
3. **Memberikan rekomendasi actionable** kepada stakeholder industri film berdasarkan hasil analisis dan modeling

### Solution Statements

Untuk mencapai tujuan yang telah ditetapkan, proyek ini mengusulkan solusi komprehensif sebagai berikut:

#### Solution Statement 1: Multi-Algorithm Approach
Mengimplementasikan dan membandingkan **tiga algoritma klasifikasi yang berbeda**:

1. **Decision Tree Classifier**:
   - **Kelebihan**: Interpretable, cepat untuk training, dapat menangani fitur kategorikal
   - **Kekurangan**: Rentan overfitting, tidak stabil terhadap perubahan data
   - **Penerapan**: Digunakan sebagai baseline model dan untuk analisis feature importance

2. **Random Forest Classifier**:
   - **Kelebihan**: Mengurangi overfitting, robust terhadap outlier, memberikan feature importance
   - **Kekurangan**: Kurang interpretable dibanding decision tree, membutuhkan lebih banyak memori
   - **Penerapan**: Diharapkan memberikan performa yang lebih stabil dibanding decision tree tunggal

3. **XGBoost Classifier**:
   - **Kelebihan**: Performa tinggi, built-in regularization, efficient computation
   - **Kekurangan**: Kompleks untuk tuning, membutuhkan pemahaman mendalam tentang parameter
   - **Penerapan**: Diharapkan memberikan performa terbaik setelah hyperparameter tuning

#### Solution Statement 2: Comprehensive Model Improvement
Melakukan **hyperparameter tuning menggunakan GridSearchCV** untuk setiap algoritma:

- **Baseline Model Training**: Melatih model dengan parameter default untuk establishing benchmark
- **Hyperparameter Optimization**: Menggunakan GridSearchCV dengan 5-fold cross-validation
- **Performance Comparison**: Membandingkan improvement antara baseline dan tuned model
- **Model Selection**: Memilih model terbaik berdasarkan multiple metrics

#### Measurable Success Criteria
Solusi akan dievaluasi menggunakan metrik-metrik berikut:
- **Akurasi**: Proporsi prediksi yang benar dari total prediksi
- **Precision**: Tingkat ketepatan prediksi positif (film sukses)
- **Recall**: Tingkat deteksi film sukses yang sebenarnya
- **F1-Score**: Harmonic mean dari precision dan recall (balanced metric)
- **AUC-ROC**: Area under curve untuk evaluasi probabilistic prediction

Target minimum performa: F1-Score ≥ 0.80 dengan AUC-ROC ≥ 0.75

## 3. Data Understanding

### Informasi Dataset

Dataset yang digunakan dalam proyek ini adalah **"Movies Recommendation Dataset"** yang diperoleh dari Kaggle dengan spesifikasi sebagai berikut:

- **Sumber Data**: [Kaggle - Movie Recommendation Dataset](https://www.kaggle.com/datasets/vyshnavi25/movie-recommendation-dataset)
- **Jumlah Observasi**: 272 film
- **Jumlah Fitur**: 21 kolom
- **Periode Data**: Film yang dirilis dalam berbagai tahun
- **Format**: CSV file dengan encoding UTF-8

### Deskripsi Variabel

Dataset mencakup 21 variabel dengan rincian sebagai berikut:

| Variabel | Tipe Data | Deskripsi | Missing Values |
|----------|-----------|-----------|----------------|
| Movie_ID | Integer | Unique identifier untuk setiap film | 0 |
| Movie_Title | Object | Judul film | 0 |
| Movie_Genre | Object | Genre film (dapat multiple) | 0 |
| Movie_Language | Object | Bahasa utama film | 0 |
| Movie_Budget | Integer | Anggaran produksi dalam USD | 0 |
| Movie_Popularity | Float | Skor popularitas (skala 0-100+) | 0 |
| Movie_Release_Date | Object | Tanggal rilis film | 0 |
| **Movie_Revenue** | **Integer** | **Pendapatan box office (Target)** | **0** |
| Movie_Runtime | Integer | Durasi film dalam menit | 0 |
| Movie_Vote | Float | Rating rata-rata (skala 1-10) | 0 |
| Movie_Vote_Count | Integer | Jumlah voting/rating | 0 |
| Movie_Homepage | Object | URL website resmi film | 151 (55.5%) |
| Movie_Keywords | Object | Kata kunci terkait film | 1 (0.4%) |
| Movie_Overview | Object | Sinopsis film | 0 |
| Movie_Production_House | Object | Studio produksi | 0 |
| Movie_Production_Country | Object | Negara produksi | 0 |
| Movie_Spoken_Language | Object | Bahasa dalam film | 0 |
| Movie_Tagline | Object | Tagline film | 15 (5.5%) |
| Movie_Cast | Object | Pemeran utama | 1 (0.4%) |
| Movie_Crew | Object | Crew film | 1 (0.4%) |
| Movie_Director | Object | Sutradara | 3 (1.1%) |

### Kondisi Data

#### Statistik Deskriptif Fitur Numerik:
```
                     Movie_Revenue    Movie_Budget    Movie_Popularity    Movie_Vote
count               272.000000       272.000000      272.000000          272.000000
mean                206,500,900      41,974,530      46.604872           7.069853
std                 259,030,000      51,013,700      35.990145           0.805545
min                 0                0               0.895946            3.500000
25%                 31,896,230       6,850,000       21.088627           6.700000
50%                 106,064,100      23,500,000      39.040441           7.100000
75%                 288,441,300      60,000,000      60.716904           7.600000
max                 1,845,034,000    300,000,000     271.972889          8.500000
```

#### Insight dari Analisis Data:
1. **Distribusi Revenue**: Sangat skewed dengan beberapa blockbuster yang mendominasi
2. **Range Budget**: Bervariasi dari film indie ($0) hingga blockbuster ($300M)
3. **Missing Values**: Movie_Homepage memiliki 55.5% missing values, yang akan dihapus
4. **Target Distribution**: Akan dibuat berdasarkan threshold $50 juta untuk klasifikasi sukses/tidak sukses

### Exploratory Data Analysis (EDA)

#### 1. Analisis Distribusi Target Variable
Berdasarkan analisis kode yang disediakan, target variable (Movie_Revenue) menunjukkan:
- **Distribusi**: Highly right-skewed dengan beberapa outlier extreme
- **Threshold Kesuksesan**: $50,000,000 (berdasarkan analisis business requirement)
- **Class Distribution**: 68% sukses (185 film) vs 32% tidak sukses (87 film)

#### 2. Analisis Korelasi Fitur
Berdasarkan correlation analysis, fitur yang paling berkorelasi dengan target:
1. **Movie_Budget** (0.412): Budget tinggi cenderung menghasilkan revenue tinggi
2. **Movie_Popularity** (0.371): Popularitas berkorelasi positif dengan kesuksesan
3. **Movie_Vote_Count** (0.367): Jumlah voting mencerminkan engagement audience

#### 3. Feature Importance Analysis
Analisis menunjukkan bahwa fitur-fitur berikut memiliki potensi prediktif:
- **Financial factors**: Budget, Popularity
- **Audience factors**: Vote count, Rating
- **Production factors**: Genre, Language, Director

## 4. Data Preparation

Tahapan data preparation yang dilakukan meliputi berbagai teknik preprocessing untuk memastikan data siap digunakan dalam modeling. Berikut adalah penjelasan detail setiap tahapan:

### 4.1 Feature Selection dan Data Cleaning

#### Penghapusan Kolom Tidak Relevan
```python
cols_to_drop = [
    'Movie_Homepage',    # 55% missing values
    'Movie_Overview',    # Text data yang sulit diproses
    'Movie_Tagline',     # Text data dengan missing values
    'Movie_Keywords',    # Text data kompleks
    'Movie_Cast',        # Text data kompleks
    'Movie_Crew',        # Text data kompleks
    'Movie_Revenue'      # Target original (mencegah data leakage)
]
```

**Alasan penghapusan**:
- **Movie_Homepage**: Memiliki 55% missing values dan kurang relevan untuk prediksi
- **Text fields**: Memerlukan preprocessing NLP yang kompleks dan di luar scope proyek ini
- **Movie_Revenue**: Dihapus dari features untuk mencegah data leakage karena ini adalah basis untuk target variable

### 4.2 Pembuatan Target Variable

#### Definisi Kesuksesan
```python
SUCCESS_THRESHOLD = 50000000  # $50 juta USD
df_processed['Success_Label'] = df['Movie_Revenue'].apply(
    lambda x: 1 if x >= SUCCESS_THRESHOLD else 0
)
```

**Alasan pemilihan threshold $50 juta**:
- Berdasarkan industry standard untuk film "commercially successful"
- Mencakup biaya produksi, marketing, dan distribusi
- Menghasilkan distribusi class yang relatif balanced (68% vs 32%)

### 4.3 Penanganan Missing Values

#### Strategi Penanganan:
```python
# Numerik: Gunakan median (robust terhadap outlier)
for col in numeric_cols:
    if df_processed[col].isnull().sum() > 0:
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)

# Kategorikal: Gunakan mode atau 'Unknown'
for col in categorical_cols:
    if df_processed[col].isnull().sum() > 0:
        mode_val = df_processed[col].mode()[0]
        df_processed[col].fillna(mode_val, inplace=True)
```

**Alasan strategi**:
- **Median untuk numerik**: Lebih robust terhadap outlier dibanding mean
- **Mode untuk kategorikal**: Mempertahankan distribusi natural dari data
- **Conservative approach**: Menghindari imputasi yang terlalu agresif

### 4.4 Feature Engineering

#### Label Encoding untuk Variabel Kategorikal
```python
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
    le_dict[col] = le
```

**Alasan menggunakan Label Encoding**:
- **Simplicity**: Sederhana dan efektif untuk tree-based algorithms
- **Memory Efficiency**: Tidak menghasilkan high-dimensional sparse matrix seperti One-Hot Encoding
- **Compatibility**: Algoritma yang digunakan (Decision Tree, Random Forest, XGBoost) dapat menangani ordinal relationship

#### Final Feature Set
Setelah preprocessing, diperoleh 13 fitur untuk modeling:
- **5 fitur numerik**: Budget, Popularity, Runtime, Vote, Vote_Count
- **8 fitur kategorikal encoded**: Title, Genre, Language, Release_Date, Production_House, Production_Country, Spoken_Language, Director

### 4.5 Data Splitting

#### Stratified Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Alasan stratification**:
- **Balanced representation**: Memastikan proporsi class yang sama di training dan testing set
- **Reliable evaluation**: Menghindari bias dalam evaluasi model
- **Reproducibility**: Random state untuk hasil yang konsisten

#### Distribusi Data:
- **Training set**: 217 observasi (69 tidak sukses, 148 sukses)
- **Test set**: 55 observasi (18 tidak sukses, 37 sukses)
- **Proporsi maintained**: ~32% tidak sukses, ~68% sukses

### 4.6 Feature Scaling

#### StandardScaler Implementation
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Alasan menggunakan StandardScaler**:
- **Algorithm requirement**: Beberapa algoritma sensitif terhadap skala fitur
- **Improved convergence**: Membantu optimasi algoritma seperti XGBoost
- **Fair comparison**: Memastikan semua fitur memiliki kontribusi yang setara

**Catatan penting**: Scaler di-fit hanya pada training data untuk mencegah data leakage.

## 5. Modeling

### 5.1 Algoritma yang Digunakan

Proyek ini mengimplementasikan tiga algoritma klasifikasi dengan karakteristik yang berbeda untuk mendapatkan solusi yang optimal:

#### 5.1.1 Decision Tree Classifier

**Konsep Dasar**:
Decision Tree adalah algoritma supervised learning yang membangun model prediksi dalam bentuk struktur pohon. Setiap node internal merepresentasikan test pada atribut, setiap branch merepresentasikan outcome dari test, dan setiap leaf node merepresentasikan class label.

**Parameter yang Digunakan**:
```python
DecisionTreeClassifier(
    criterion='entropy',        # Measure untuk information gain
    max_depth=3,               # Kedalaman maksimum pohon
    min_samples_split=2,       # Minimum sampel untuk split
    min_samples_leaf=1,        # Minimum sampel di leaf node
    random_state=42
)
```

**Kelebihan**:
- **High Interpretability**: Mudah dipahami dan divisualisasikan
- **No Assumptions**: Tidak memerlukan asumsi tentang distribusi data
- **Feature Selection**: Built-in feature selection melalui splitting criteria
- **Fast Training**: Relatif cepat untuk training
- **Handle Mixed Data**: Dapat menangani fitur numerik dan kategorikal

**Kekurangan**:
- **Overfitting Prone**: Mudah mengalami overfitting, terutama dengan data kecil
- **Instability**: Sensitif terhadap perubahan kecil dalam data
- **Bias**: Cenderung bias pada fitur dengan banyak level
- **Poor Generalization**: Sulit generalisasi pada data yang sangat berbeda

**Penerapan dalam Proyek**:
Digunakan sebagai baseline model dan untuk analisis interpretability. Parameter optimal ditemukan melalui grid search dengan fokus pada pencegahan overfitting.

#### 5.1.2 Random Forest Classifier

**Konsep Dasar**:
Random Forest adalah ensemble method yang mengkombinasikan multiple decision trees dengan teknik bootstrap aggregating (bagging). Setiap tree dilatih pada subset data yang berbeda dan menggunakan subset fitur yang random.

**Parameter yang Digunakan**:
```python
RandomForestClassifier(
    n_estimators=100,          # Jumlah trees dalam forest
    max_depth=5,               # Kedalaman maksimum setiap tree
    min_samples_split=5,       # Minimum sampel untuk split
    min_samples_leaf=2,        # Minimum sampel di leaf
    random_state=42,
    n_jobs=-1                  # Parallel processing
)
```

**Kelebihan**:
- **Reduced Overfitting**: Bagging mengurangi variance dan overfitting
- **Feature Importance**: Memberikan ranking importance fitur
- **Robust to Outliers**: Lebih tahan terhadap outlier dibanding single tree
- **Good Performance**: Umumnya memberikan performa yang baik tanpa tuning ekstensif
- **Handle Missing Values**: Dapat menangani missing values secara internal

**Kekurangan**:
- **Less Interpretable**: Lebih sulit diinterpretasi dibanding single tree
- **Memory Intensive**: Membutuhkan lebih banyak memori untuk menyimpan multiple trees
- **Potential Bias**: Masih dapat bias pada fitur yang dominan
- **Slower Prediction**: Prediksi lebih lambat karena ensemble voting

**Penerapan dalam Proyek**:
Diharapkan memberikan performa yang lebih stabil dan robust dibanding Decision Tree tunggal, dengan tetap mempertahankan interpretability melalui feature importance.

#### 5.1.3 XGBoost Classifier

**Konsep Dasar**:
XGBoost (Extreme Gradient Boosting) adalah implementasi advanced dari gradient boosting algorithm. Berbeda dengan Random Forest yang menggunakan bagging, XGBoost menggunakan boosting dimana setiap tree belajar dari kesalahan tree sebelumnya.

**Parameter yang Digunakan**:
```python
XGBClassifier(
    n_estimators=100,          # Jumlah boosting rounds
    max_depth=3,               # Kedalaman maksimum tree
    learning_rate=0.01,        # Step size shrinkage
    subsample=0.8,             # Subsample ratio
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
```

**Kelebihan**:
- **Superior Performance**: Umumnya memberikan performa terbaik dalam kompetisi ML
- **Built-in Regularization**: L1 dan L2 regularization untuk mencegah overfitting
- **Handle Missing Values**: Built-in handling untuk missing values
- **Feature Importance**: Menyediakan multiple jenis feature importance
- **Efficient Implementation**: Optimasi untuk kecepatan dan memori

**Kekurangan**:
- **Complex Tuning**: Banyak hyperparameter yang perlu di-tune
- **Less Interpretable**: Sulit untuk diinterpretasi dibanding tree-based method lainnya
- **Prone to Overfitting**: Tanpa regularization yang tepat, mudah overfit
- **Computational Intensive**: Membutuhkan computational resource yang lebih besar

**Penerapan dalam Proyek**:
Diharapkan memberikan performa terbaik setelah hyperparameter tuning, terutama untuk kasus dengan fitur yang kompleks dan heterogen.

### 5.2 Hyperparameter Tuning Strategy

#### Grid Search Configuration
Untuk setiap algoritma, dilakukan systematic search terhadap kombinasi parameter terbaik:

```python
models_config = {
    'Decision Tree': {
        'params': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    },
    'Random Forest': {
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'XGBoost': {
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
}
```

#### Cross-Validation Strategy
- **5-Fold Stratified CV**: Memastikan distribusi class yang konsisten
- **Scoring Metric**: F1-score dipilih sebagai primary metric karena balanced importance antara precision dan recall
- **Reproducibility**: Random state ditetapkan untuk hasil yang konsisten

### 5.3 Model Training Process

#### Baseline vs Tuned Model Comparison
Untuk setiap algoritma, dilakukan dua tahap training:

1. **Baseline Training**: Model dengan parameter default
2. **Hyperparameter Tuning**: GridSearchCV untuk optimasi parameter
3. **Performance Comparison**: Evaluasi improvement dari tuning

#### Training Results Summary:

| Model | Baseline F1 | Tuned F1 | Improvement | Best Parameters |
|-------|-------------|----------|-------------|-----------------|
| Decision Tree | 0.8250 | 0.8706 | +0.0456 (+5.5%) | criterion='entropy', max_depth=3 |
| Random Forest | 0.9024 | 0.8916 | -0.0109 (-1.2%) | max_depth=5, min_samples_leaf=2 |
| XGBoost | 0.8675 | 0.8222 | -0.0452 (-5.2%) | learning_rate=0.01, max_depth=3 |

### 5.4 Model Selection

#### Selection Criteria
Berdasarkan evaluasi comprehensive menggunakan multiple metrics:

1. **Primary Metric**: F1-Score (balanced precision-recall)
2. **Secondary Metrics**: AUC-ROC, Accuracy, Cross-validation stability
3. **Practical Considerations**: Overfitting analysis, interpretability

#### Final Model Selection: **Random Forest**
- **F1-Score**: 0.8916 (tertinggi)
- **AUC-ROC**: 0.8393 (baik)
- **Cross-validation**: Stable performance (CV std: 0.0402)
- **Interpretability**: Good balance antara performa dan interpretability
- **Robustness**: Menunjukkan generalisasi yang baik

**Alasan Pemilihan**:
Random Forest dipilih sebagai model final karena memberikan kombinasi terbaik antara performa tinggi, stabilitas, dan interpretability, meskipun mengalami sedikit penurunan dari baseline (masih dalam acceptable range).

## 6. Evaluation

### 6.1 Metrik Evaluasi yang Digunakan

Proyek ini menggunakan multiple metrics untuk evaluasi comprehensive sesuai dengan konteks problem klasifikasi binary:

#### 6.1.1 Accuracy
**Formula**: 
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretasi**: 
Proporsi prediksi yang benar dari total prediksi. Dalam konteks prediksi kesuksesan film, accuracy menunjukkan seberapa sering model dapat memprediksi dengan benar apakah sebuah film akan sukses atau tidak.

**Relevansi**: 
Meskipun accuracy adalah metrik yang intuitif, dalam konteks business ini tidak menjadi prioritas utama karena cost dari false negative (memprediksi film tidak sukses padahal sukses) dan false positive (memprediksi film sukses padahal tidak) memiliki implikasi bisnis yang berbeda.

#### 6.1.2 Precision
**Formula**: 
```
Precision = TP / (TP + FP)
```

**Interpretasi**: 
Dari semua film yang diprediksi sukses, berapa persen yang benar-benar sukses. Precision tinggi berarti model konservatif dalam memprediksi kesuksesan.

**Relevansi Bisnis**: 
Precision penting untuk investor dan studio karena false positive berarti investasi pada film yang sebenarnya tidak akan sukses, yang dapat mengakibatkan kerugian finansial yang signifikan.

#### 6.1.3 Recall (Sensitivity)
**Formula**: 
```
Recall = TP / (TP + FN)
```

**Interpretasi**: 
Dari semua film yang benar-benar sukses, berapa persen yang berhasil diprediksi oleh model. Recall tinggi berarti model sensitif dalam mendeteksi film sukses.

**Relevansi Bisnis**: 
Recall penting untuk memastikan tidak melewatkan opportunity. False negative berarti kehilangan kesempatan untuk berinvestasi pada film yang sebenarnya akan sukses.

#### 6.1.4 F1-Score
**Formula**: 
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**Interpretasi**: 
Harmonic mean dari precision dan recall, memberikan balanced measure ketika kedua metrik sama pentingnya.

**Mengapa Dipilih sebagai Primary Metric**: 
Dalam konteks prediksi kesuksesan film, baik precision maupun recall sama-sama penting. Studio membutuhkan prediksi yang akurat (precision tinggi) namun juga tidak ingin melewatkan opportunity (recall tinggi). F1-score memberikan balance optimal antara kedua aspek ini.

#### 6.1.5 AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
**Formula**: 
AUC adalah area di bawah kurva ROC, dimana ROC curve menunjukkan trade-off antara True Positive Rate (Recall) dan False Positive Rate.

**Interpretasi**: 
AUC-ROC mengukur kemampuan model untuk membedakan antara kedua class di semua threshold. Nilai 0.5 menunjukkan performa random, sedangkan 1.0 menunjukkan perfect classifier.

**Relevansi**: 
AUC-ROC penting karena memberikan gambaran performa model secara keseluruhan, tidak tergantung pada threshold tertentu. Ini berguna untuk stakeholder yang mungkin ingin menyesuaikan threshold berdasarkan risk tolerance mereka.

### 6.2 Hasil Evaluasi Model

#### 6.2.1 Performa Final Setiap Model

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | CV Mean | CV Std |
|-------|----------|-----------|--------|----------|---------|---------|---------|
| **Random Forest** | **0.8364** | **0.8043** | **1.0000** | **0.8916** | **0.8393** | **0.8864** | **0.0402** |
| Decision Tree | 0.8000 | 0.7708 | 1.0000 | 0.8706 | 0.8086 | 0.8543 | 0.0503 |
| XGBoost | 0.7091 | 0.6981 | 1.0000 | 0.8222 | 0.8168 |
6.3 Model Terbaik

Berdasarkan seluruh metrik evaluasi, model Random Forest memiliki performa terbaik secara keseluruhan. Model ini memiliki keseimbangan antara akurasi, precision, recall, dan F1-score yang sangat baik, serta nilai AUC-ROC yang tinggi dan hasil validasi silang (cross-validation) yang stabil.

Model ini direkomendasikan sebagai model akhir untuk digunakan dalam sistem prediksi keberhasilan film.

7. Kesimpulan

Model klasifikasi berhasil dibangun untuk memprediksi keberhasilan film berdasarkan data historis. Dengan evaluasi menyeluruh dan penggunaan beberapa algoritma, proyek ini menunjukkan bahwa prediksi sukses film bisa dilakukan secara cukup akurat dengan data yang tersedia.

Model terbaik yang dipilih adalah Random Forest, dengan performa unggul secara konsisten di berbagai metrik.
