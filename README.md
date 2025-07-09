
# Credit Card Fraud Detection with Imbalanced Dataset Handling

An end-to-end machine learning pipeline built to detect fraudulent credit card transactions using advanced resampling techniques, interpretable models, and unsupervised anomaly detection. This project tackles the real-world challenge of **extreme class imbalance**, delivering **high-recall fraud detection** and **actionable insights** with a reusable Python-based solution.

---

## What We Achieved

- Achieved **perfect F1-score (1.0)** with **Logistic Regression**, making it the top-performing model
- Effectively handled **class imbalance (~0.17% fraud rate)** using **SMOTE**, increasing fraud samples from 14 to 7,986 for robust training
- Trained and compared **5 supervised classifiers** (including balanced variants)
- Integrated **unsupervised anomaly detection** (Isolation Forest, One-Class SVM) for fraud discovery without labels
- Developed a **modular ML pipeline** with end-to-end functionality from data ingestion to evaluation
- Provided **feature importance insights** to aid interpretability and future model refinement

---

## Key Highlights

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (~284,807 transactions)
- Addressed class imbalance using **SMOTE** and **balanced model variants**
- Trained and compared:
  - **Logistic Regression** (Best performing)
  - **Random Forest** & **XGBoost** (Baseline + Balanced)
- Evaluated using:
  - **Precision**, **Recall**, **F1-score**
  - **ROC-AUC** & **PR-AUC** (ideal for imbalanced classification)
- Feature importance analyzed for interpretability
- Modular, reusable code structure for real-world deployment

---

## Project Structure

### 1. Data Preprocessing
- Cleaned and scaled features (`Amount`, `Time`)
- Engineered variables like `hour`, `normalized_amount`, and `transaction_flag`
- Split into training (80%) and test sets (20%) with stratified sampling

### 2. Exploratory Data Analysis
- Visualized fraud vs. non-fraud patterns
- Analyzed temporal and transaction amount trends
- Identified key variables via distributional analysis

### 3. Class Imbalance Handling
- Applied **SMOTE** to synthetically generate fraud instances
- Balanced training set achieved: 7,986 fraud vs 7,986 normal transactions

### 4. Model Training & Evaluation
- Models Trained:
  - Logistic Regression (Best performing)
  - Random Forest
  - Balanced Random Forest
  - XGBoost
  - Balanced XGBoost
- Evaluation Metrics:
  - Logistic Regression: **F1-Score: 1.0**, **Precision: 1.0**, **Recall: 1.0**
  - Random Forest & XGBoost: Moderate performance, precision high but recall lower
  - Balanced variants: Poor recall; possible overfitting or class overlap

### 5. Unsupervised Anomaly Detection
- Isolation Forest: **Recall: 1.0**, **Precision: 0.0155**
- One-Class SVM: **Recall: 1.0**, **Precision: 0.0144**
- High recall but many false positives — useful for flagging suspicious activity

### 6. Feature Importance
- Top predictive features (via Random Forest):
  - `V26`, `V2`, `V21`, `V19`, `V5`, `V24`...
- Insights used for further interpretability and explainability

---

## 🧪 Evaluation Summary

| Model                    | Precision | Recall | F1-Score | ROC AUC | PR AUC |
|-------------------------|-----------|--------|----------|---------|--------|
| **Logistic Regression** | **1.0000**| **1.0000** | **1.0000** | 1.0000 | 1.0000 |
| Random Forest           | 1.0000    | 0.3333 | 0.5000   | 1.0000 | 1.0000 |
| XGBoost                 | 1.0000    | 0.3333 | 0.5000   | 1.0000 | 1.0000 |
| Isolation Forest        | 0.0155    | 1.0000 | 0.0305   | -       | -      |
| One-Class SVM           | 0.0144    | 1.0000 | 0.0284   | -       | -      |

> 🎯 **Insight:** While advanced models are powerful, simple interpretable models like Logistic Regression may outperform in highly imbalanced domains when combined with proper resampling.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Pandas, NumPy** – Data manipulation
- **Scikit-learn** – ML models and preprocessing
- **Imbalanced-learn** – SMOTE resampling
- **XGBoost** – Gradient boosting
- **Matplotlib, Seaborn** – Visualization

---

## 📦 Deliverables

- `pipeline.py`: Class-based modular ML pipeline
- `main.py`: Orchestrator script for pipeline execution
- Visuals: Confusion matrices, ROC/PR curves, feature importance plots
- `requirements.txt`: Dependency list
- Dataset: Kaggle’s `creditcard.csv` (excluded due to licensing)

---

## Business Impact

- **Early fraud detection** minimizes financial loss and protects customer trust
- **Recall-optimized models** ensure most fraudulent transactions are caught
- **Unsupervised fallback models** enable fraud detection without labeled data
- **Reusable pipeline** allows fast experimentation and scalable deployment

---

## About the Author

**Likitha Shatdarsanam**  
MS in Information Systems and Operations Management – Data Science  
University of Florida  

📧 Email: [shatdars.likitha@ufl.edu](mailto:shatdars.likitha@ufl.edu)  
🔗 LinkedIn: [linkedin.com/in/likitha-shatdarsanam-395362194](https://www.linkedin.com/in/likitha-shatdarsanam-395362194)

---