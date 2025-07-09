
# 🔐 Credit Card Fraud Detection with Imbalanced Dataset Handling

An end-to-end machine learning project focused on identifying fraudulent credit card transactions using advanced resampling techniques and classification models. This solution addresses the real-world challenge of class imbalance while maximizing detection performance using interpretable and scalable methods.

---

## ✅ Key Highlights

- Utilized the popular **Kaggle Credit Card Fraud dataset** (~284,807 transactions)
- Performed **data cleaning**, transformation, and **feature engineering**
- Applied **SMOTE** to counter extreme class imbalance (~0.17% fraud)
- Built and compared multiple machine learning models:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- Evaluated models using **precision**, **recall**, **F1-score**, **ROC-AUC**, and **PR-AUC**
- Incorporated **unsupervised anomaly detection** with Isolation Forest and One-Class SVM
- Visualized performance metrics and **feature importance**
- Packaged into a **modular, reusable Python pipeline**

---

## 🧱 Key Components

### 1. Data Cleaning & Preparation
- Removed nulls and irrelevant columns
- Engineered features such as `hour`, `normalized_amount`, and `transaction_flag`
- Scaled `Amount` and `Time` using `StandardScaler`

### 2. Exploratory Data Analysis (EDA)
- Analyzed fraud vs. non-fraud distributions
- Visualized feature relationships using **Seaborn**, **Matplotlib**
- Identified high-risk patterns for time-based and amount-based fraud

### 3. Model Building
- Trained supervised classifiers:
  - **Logistic Regression** for baseline interpretability
  - **Random Forest** for ensemble-based robustness
  - **XGBoost** for high-performance gradient boosting
- Included **SMOTE resampling** in the pipeline for minority class balancing

### 4. Anomaly Detection
- Used **Isolation Forest** and **One-Class SVM** to detect fraud in an unsupervised setting
- Compared anomaly scores with known fraud labels

### 5. Model Evaluation
- **Precision**: Avoid false accusations
- **Recall**: Catch as many frauds as possible
- **F1-score**: Balanced metric for skewed datasets
- **AUC-ROC & PR-AUC**: Threshold-independent evaluation

---

## 📁 Deliverables

- `pipeline.py`: Class-based reusable ML pipeline
- `main.py`: Executable script for end-to-end fraud detection
- `creditcard.csv`: Kaggle dataset (not included for licensing reasons)
- `requirements.txt`: All Python dependencies
- Visual outputs: Confusion matrix, ROC curves, PR curves, feature importance charts

---

## ⚙️ Tools & Technologies

- **Python**: Core programming language
- **Pandas, NumPy**: Data manipulation
- **Scikit-learn**: ML models, metrics, and preprocessing
- **Imbalanced-learn**: SMOTE and other resampling techniques
- **XGBoost**: Advanced classification
- **Matplotlib, Seaborn**: Data visualization

---

## 📈 Business Relevance

- **Fraud detection** is critical for financial security and minimizing losses
- Realistic handling of **class imbalance** simulates actual deployment scenarios
- Models like **Random Forest** and **XGBoost** offer high recall with acceptable precision
- **Anomaly detection** provides a fallback when labels are not available

---

## 👩‍💻 About the Author

**Likitha Shatdarsanam**  
MS in Information Systems and Operations Management – Data Science  
University of Florida  

📧 Email: [shatdars.likitha@ufl.edu](mailto:shatdars.likitha@ufl.edu)  
🔗 LinkedIn: [linkedin.com/in/likitha-shatdarsanam-395362194](https://www.linkedin.com/in/likitha-shatdarsanam-395362194)
