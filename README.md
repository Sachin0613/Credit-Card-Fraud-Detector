# 💳 Credit Card Fraud Detection

![Project Status](https://img.shields.io/badge/status-completed-brightgreen)
![Machine Learning](https://img.shields.io/badge/tech-ML-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 📽️ Demo Video
Watch the full project walkthrough on LinkedIn:  
🔗 [Click here to watch](https://www.linkedin.com/posts/sachin-yadav-631b6031a_neuronexus-internship-webdevelopment-activity-7324427848058834944-Nw_1?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFD5DYUBVRcx5EYX9IkvTniVPLgN2CcJxwI)

---

## 📌 Overview

This project aims to **detect fraudulent credit card transactions** using supervised machine learning models. It tackles the challenge of working with an **imbalanced dataset** by using advanced techniques like **SMOTE** for oversampling.

---

## 🧠 Techniques & Tools

- Logistic Regression
- Random Forest Classifier
- SMOTE (Synthetic Minority Oversampling Technique)
- Classification Report
- Confusion Matrix
- ROC Curve & AUC Score

---

## 📊 Dataset

- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Classes**: 0 = Legitimate, 1 = Fraudulent
- Highly imbalanced dataset (~0.17% fraud cases)

---

## ⚙️ Project Workflow

1. **Data Preprocessing**
   - Null check
   - Standardization
   - Splitting into training and test sets

2. **Handling Imbalanced Data**
   - Applied SMOTE to oversample the minority class (fraud)

3. **Model Building**
   - Trained Logistic Regression and Random Forest
   - Compared performance using metrics

4. **Model Evaluation**
   - Classification report (Precision, Recall, F1)
   - Confusion matrix visualization
   - ROC-AUC curve

---

## 📈 Results

| Model               | Accuracy | Precision | Recall | AUC  |
|--------------------|----------|-----------|--------|------|
| Logistic Regression| ~98.8%   | High      | Moderate | High |
| Random Forest       | ~99.2%   | High      | High    | Very High |

