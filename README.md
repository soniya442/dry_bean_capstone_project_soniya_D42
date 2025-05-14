# 🫘 Dry Bean Classification using Machine Learning

This project focuses on classifying different types of **Dry Beans** based on their morphological features using machine learning models. It uses a clean end-to-end pipeline with preprocessing, feature selection, SMOTE, scaling, and hyperparameter tuning — implemented using **Logistic Regression** and other classifiers.

---

## 📊 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)
- **Records**: 13,611 samples
- **Target Variable**: `Class` (Bean Type: e.g., SIRA, BOMBAY, DERMASON, etc.)
- **Features**: 16 numerical features representing shape and size metrics (e.g., Area, Perimeter, Aspect Ratio)

---

## 🔍 Objective

> To classify dry beans into their correct type based on their physical measurements using machine learning.

---

## ✅ Project Workflow

### 1. 📦 Data Preprocessing
- Removed duplicates
- Outlier handling using **IQR method**
- **Skewness correction** using `PowerTransformer` (Yeo-Johnson)
- Feature Scaling with `StandardScaler`

### 2. 🧠 Feature Selection
- Removed highly correlated features (multicollinearity)
- Evaluated importance using **ANOVA F-test**
- Selected top 6 features for modeling

### 3. ⚖️ Class Imbalance Handling
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) on the training set

### 4. 🔁 Train-Test Split
- 80% Train / 20% Test split after feature selection

### 5. 🔗 ML Pipeline
- Built a **Scikit-learn Pipeline** including:
  - Imputation
  - Yeo-Johnson transformation
  - Scaling
  - SMOTE
  - Classifier

### 6. 🔧 Hyperparameter Tuning
- Used **GridSearchCV** inside the pipeline
- Tuned Logistic Regression (`C`, `solver`)

### 7. 📈 Model Evaluation
- Evaluated using accuracy, classification report, and confusion matrix
- Compared Logistic Regression, KNN, SVM, Decision Tree, Random Forest

---

## 📊 Final Model Results

| Metric       | Score  |
|--------------|--------|
| Train Accuracy | 90.7% |
| Test Accuracy  | 88.2% |
| F1-score (macro avg) | 89% |
| Best Model    | Logistic Regression (with tuning) |

---

## 📁 Files Included

- `drybean_pipeline_tuned_logreg.pkl` – saved pipeline model
- `new_bean_data.csv` – sample input data for prediction
- `drybean_notebook.ipynb` – full code notebook (Colab)

---

## 🚀 How to Use

1. Clone this repo or open the notebook in Google Colab  
2. Run all cells for EDA, training, and evaluation  
3. Load `drybean_pipeline_tuned_logreg.pkl` for prediction on new data

```python
import joblib
model = joblib.load('drybean_pipeline_tuned_logreg.pkl')
new_data = pd.read_csv('new_bean_data.csv')
print(model.predict(new_data))

