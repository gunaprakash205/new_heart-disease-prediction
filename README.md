# new_heart-disease-prediction
Heart Disease Risk Prediction using Machine Learning and SHAP Explainable AI, with a Streamlit web application for interactive predictions and model interpretation.


# ❤️ Heart Disease Risk Prediction

An end-to-end **Machine Learning project** that predicts heart disease risk using patient health information. The project compares multiple classification algorithms, selects a Random Forest model based on test-set performance, and uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

The model is deployed as an interactive **Streamlit web application**.

## 🚀 Live Demo

**Streamlit App:**

https://newheart-disease-prediction-sdcdkisyyrme36ny53nyrm.streamlit.app/

> ⚠️ This project is for educational purposes only and is not intended for medical diagnosis or clinical decision-making.

---

## 📌 Project Overview

The goal of this project is to build a machine-learning classification system that can identify whether a patient is predicted to have heart disease based on selected health parameters.

The project follows an end-to-end ML workflow:


Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Categorical Encoding
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Multiple ML Models
   ↓
Model Evaluation
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Random Forest Selection
   ↓
SHAP Explainability
   ↓
Streamlit Deployment


---

## 🧠 Machine Learning Models

The following classification algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

### Model Performance

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     84.24% |     84.11% |     88.24% |     86.12% |
| Decision Tree       |     79.35% |     80.19% |     83.33% |     81.73% |
| **Random Forest**   | **86.96%** | **84.82%** | **93.14%** | **88.79%** |
| Gradient Boosting   |     82.61% |     83.02% |     86.27% |     84.62% |
| SVM                 |     85.33% |     83.19% |     92.16% |     87.44% |
| KNN                 |     85.33% |     85.05% |     89.22% |     87.08% |

### 🏆 Selected Model

**Random Forest** achieved the strongest performance on the held-out test set:

* Accuracy: **86.96%**
* Precision: **84.82%**
* Recall: **93.14%**
* F1 Score: **88.79%**

The model achieved particularly high recall for the positive class, which is an important metric for this classification task.

---

## 🔄 Cross-Validation

The Random Forest model was also evaluated using **5-fold cross-validation**.

### Cross-Validation Accuracy


Fold 1: 81.76%
Fold 2: 83.67%
Fold 3: 80.27%
Fold 4: 78.23%
Fold 5: 82.99%

Average: 81.39%


### Cross-Validation Recall


Fold 1: 84.15%
Fold 2: 92.68%
Fold 3: 81.48%
Fold 4: 83.95%
Fold 5: 86.42%

Average: 85.74%


Cross-validation was used to evaluate how consistently the model performs across different subsets of the dataset.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

### Missing Values

* Numerical missing values were handled using the **median**.
* Categorical missing values were handled using the **mode**.

### Categorical Encoding

Categorical variables were converted into numerical features using **one-hot encoding**.


pd.get_dummies(
    data,
    columns=cat_cols,
    drop_first=True
)


### Feature Scaling

Numerical features were scaled before model training using a scaler.

The same fitted scaler is used when processing new patient data in the Streamlit application.

---

## 🔍 Explainable AI — SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to understand individual model predictions.

Instead of only displaying:


Prediction → Heart Disease


SHAP helps answer:

> **Why did the model make this prediction?**

For each prediction, SHAP identifies features that contributed toward:

* Class 0 — No Heart Disease
* Class 1 — Heart Disease

### SHAP Interpretation


Positive SHAP value
        ↓
Pushes prediction toward Class 1

Negative SHAP value
        ↓
Pushes prediction toward Class 0

Larger absolute SHAP value
        ↓
Greater influence on the prediction


The Streamlit application displays the most influential features for the individual prediction.

> SHAP explains the behavior of the machine-learning model. It does not establish medical causation.

---

## 🌐 Streamlit Application

The trained model is deployed using **Streamlit**.

The application allows users to enter patient information such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Resting ECG
* Maximum heart rate
* Exercise-induced angina
* Oldpeak
* Slope
* Number of major vessels
* Thalassemia

The application then provides:

* Model prediction
* Class probabilities
* Prediction confidence
* Patient input summary
* SHAP-based explanation
* Model performance
* Algorithm comparison

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **SHAP**
* **Streamlit**
* **Pickle**
* **Jupyter Notebook**

---

## 📂 Project Structure


Heart-Disease-Prediction/
│
├── app1.py
├── heart_model.pkl
├── scaler.pkl
├── notebook.ipynb
├── README.md
└── requirements.txt



## 📦 Requirements

Example `requirements.txt`:

```text
numpy
pandas
scikit-learn
matplotlib
shap
streamlit


---

## 📊 Key Learning Outcomes

Through this project, I worked with:

* Data preprocessing
* Missing-value treatment
* Categorical encoding
* Feature scaling
* Classification algorithms
* Model comparison
* Accuracy, precision, recall and F1-score
* Cross-validation
* Hyperparameter tuning
* Random Forest
* SHAP Explainable AI
* Model serialization using Pickle
* Streamlit application development
* ML model deployment

---

## ⚠️ Disclaimer

This application is a **machine-learning demonstration project**.

The predictions and probabilities generated by the model should **not** be interpreted as medical diagnoses, clinical risk scores, or treatment recommendations.

Always consult a qualified healthcare professional for medical evaluation.

---

## 👨‍💻 Author

**Guna Prakash**

Built as a Machine Learning and Explainable AI portfolio project.
