#Metadata
title: Customer Churn Predictor
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false

# 🧠 Customer Churn Predictor – Gradio App

A machine learning web app to predict customer churn using the Telco Customer Churn dataset. Trained using XGBoost and deployed with Gradio on Hugging Face Spaces.

## 🚀 Demo

Enter customer details to predict the likelihood of churn. The model analyzes usage behavior, contract type, billing preferences, and more to estimate the risk of a customer leaving.

## 📂 How It Works

- Preprocessed the Telco dataset (cleaning, encoding, scaling)
- Trained multiple models: Logistic Regression, Random Forest, XGBoost
- Tuned hyperparameters for best performance (XGBoost selected)
- Saved model and required metadata with joblib
- Built a Gradio UI for real-time inference
- Deployed to Hugging Face Spaces for public use

## 📈 Example Inputs

| Feature            | Type       | Example Value         |
|--------------------|------------|------------------------|
| SeniorCitizen      | Binary     | 0                     |
| Tenure             | Numeric    | 12                    |
| MonthlyCharges     | Numeric    | 79.5                  |
| TotalCharges       | Numeric    | 945.3                 |
| Contract           | Categorical| Month-to-month        |
| InternetService    | Categorical| Fiber optic           |
| PaymentMethod      | Categorical| Electronic check      |

## 🧪 Model Info

- **Algorithm**: XGBoost Classifier
- **Accuracy**: ~84%
- **Preprocessing**: One-hot encoding, StandardScaler

## 📦 Dependencies

See `requirements.txt` in the repo.

## 🙋 Author

**Abhishek Singh**  
Research Analyst & ML Enthusiast  
[GitHub](https://github.com/100abhishek) | [LinkedIn](https://www.linkedin.com/in/abhisheksingh100/)  

---

### 🚧 Note

This app is for educational/demo purposes using open data from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).
