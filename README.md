# 🧠 HR Employee Attrition Prediction

A complete end-to-end Machine Learning project that predicts whether an
employee is likely to leave a company, built using the IBM HR Analytics
dataset from Kaggle.

---

## 📌 Problem Statement

Employee attrition is one of the most costly challenges for organizations.
Companies spend 15–20% of an employee's salary to recruit a replacement,
and it takes an average of 52 days to fill a position. This project builds
a classification model to proactively identify employees at risk of leaving,
enabling HR teams to take early retention action.

---

## 📂 Dataset

- **Source:** [IBM HR Analytics Attrition Dataset – Kaggle](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size:** 1,470 employees × 35 features
- **Target variable:** `Attrition` (1 = Left, 0 = Stayed)
- **Class distribution:** 83.9% stayed, 16.1% left (imbalanced dataset)
- Dataset is loaded directly from a local CSV file upload — no Drive
  mounting required.

---

## 🔧 Tech Stack

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## 🗂️ Project Structure

hr-attrition-prediction/
│
├── HR_Attrition_Prediction.ipynb   # Main Colab notebook
├── HR_Employee_Attrition.csv       # Dataset (uploaded locally)
└── README.md

---

## 🚀 Workflow

### Task 1 – Business Understanding
Defined the problem and justified the need for a predictive attrition
model based on real-world hiring cost statistics.

### Task 2 – Import Libraries and Dataset
Loaded the dataset directly from a local PC upload using Pandas.
Dataset contains 35 features with zero missing values.

### Task 3 – Exploratory Data Analysis
- Encoded binary columns (Attrition, OverTime, Over18) to integers
- Confirmed no missing values via heatmap
- Plotted feature distributions — identified right-skewed features
  (MonthlyIncome, TotalWorkingYears) and zero-variance columns
- Compared statistics between employees who left vs stayed
- Built correlation heatmap — strong links found between JobLevel,
  MonthlyIncome, and TotalWorkingYears
- KDE plots for DistanceFromHome, YearsWithCurrManager, TotalWorkingYears
- Box plots for MonthlyIncome by Gender and JobRole

**Key EDA Insights:**
- Single employees leave more than married/divorced ones
- Sales Representatives have the highest attrition rate
- Employees who left were younger and lived farther from home
- Low JobInvolvement and low JobLevel strongly correlate with attrition

### Task 4 – Data Preprocessing
- Dropped irrelevant columns: EmployeeCount, StandardHours, Over18,
  EmployeeNumber
- One-Hot Encoded 6 categorical columns → 26 binary features
- Concatenated with 24 numerical features → 50 total features
- Applied MinMaxScaler to normalize all values to [0, 1]

### Task 5 – Model Theory
Covered intuition behind three classifiers:
- **Logistic Regression** – sigmoid function applied to linear equation
- **Artificial Neural Networks (ANN)** – weighted layers with sigmoid
  activation
- **Random Forest** – ensemble of decision trees with majority voting

### Task 6 – Evaluation Metrics
Given class imbalance, accuracy alone is misleading. Key metrics used:
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN) ← most critical for this problem
- **F1-Score** = harmonic mean of Precision and Recall
- **Confusion Matrix** for visual performance breakdown

### Task 7 – Logistic Regression Results

| Metric | Class 0 (Stayed) | Class 1 (Left) |
|--------|-----------------|----------------|
| Precision | 0.91 | 0.83 |
| Recall | 0.98 | 0.43 |
| F1-Score | 0.94 | 0.56 |

- **Overall Accuracy: ~89.9%**
- The model performs strongly on the majority class but struggles to
  catch actual leavers (recall = 0.43), which is the core limitation.

---

## ⚠️ Key Limitation

The dataset is imbalanced (84% stayed vs 16% left). The Logistic
Regression model catches fewer than half of actual attrition cases.
Future improvements could include SMOTE oversampling, class weighting,
or switching to Random Forest / ANN to improve minority class recall.

---

## 📊 Results Summary

The Logistic Regression classifier achieves ~90% overall accuracy but
reveals the classic imbalanced-data trap — high accuracy masking poor
recall on the minority class. This project demonstrates why F1-Score
and Recall are the right KPIs for attrition prediction, not accuracy.

---

## 🙋 Author

**PRASHANT KUMAR**  
Email : pj.prashant95@gmail.com
LinkedIn : www.linkedin.com/in/prashant-kumar-a5a616298 
