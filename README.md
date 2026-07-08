# Credit Risk Prediction using Machine Learning

## Overview

This project predicts whether a loan applicant has a **Good** or **Bad** credit risk using Machine Learning. Multiple classification algorithms were trained and compared, and the best-performing model was selected for deployment.

An interactive web application was built using Streamlit, allowing users to enter applicant information and receive instant credit risk predictions.

---

## Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Label Encoding of categorical variables
- Model training and hyperparameter tuning
- Comparison of multiple ML algorithms
- Best model selection based on accuracy
- Model serialization using Joblib
- Streamlit web application for prediction

---

## Dataset

The project uses the German Credit Risk dataset containing applicant information such as:

- Age
- Sex
- Job
- Housing
- Saving Accounts
- Checking Account
- Credit Amount
- Loan Duration
- Credit Risk

---

## Machine Learning Models Used

- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier
- XGBoost Classifier

---

## Model Performance

| Model | Accuracy |
|--------|----------|
| Decision Tree | 58.10% |
| Random Forest | 61.90% |
| Extra Trees | 64.76% |
| XGBoost | 68.57% |

**Best Model:** XGBoost Classifier

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

---

## Project Structure

```
Credit_Risk_Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── credit_risk_prediction.ipynb
├── extra_trees_credit_model.pkl
├── Sex_encoder.pkl
├── Housing_encoder.pkl
├── Saving accounts_encoder.pkl
├── Checking account_encoder.pkl
├── Output Images/
└── Dataset/
```

---

## How to Run

Clone the repository

```bash
git clone <repository-link>
```

Move to the project directory

```bash
cd Credit_Risk_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## Output

The application predicts whether the applicant's credit risk is:

- Good
- Bad

---

## Future Improvements

- Improve model accuracy using advanced feature engineering
- Deploy the application online
- Add probability scores for predictions
- Enhance the user interface

---

## Author

Lakshay Kapoor
