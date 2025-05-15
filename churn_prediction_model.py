# app.py

import pandas as pd
import joblib
import gradio as gr

# Load saved objects (make sure these files are in the same directory as app.py)
feature_columns = joblib.load('feature_columns.pkl')
num_cols = joblib.load('num_cols.pkl')
scaler = joblib.load('scaler.pkl')
best_model = joblib.load('best_model.pkl')

def predict_churn(SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
                  gender, Partner, Dependents, PhoneService, MultipleLines,
                  InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                  TechSupport, StreamingTV, StreamingMovies, Contract,
                  PaperlessBilling, PaymentMethod):
    try:
        # Prepare input data as a dictionary
        input_data = {
            "SeniorCitizen": SeniorCitizen,
            "tenure": float(tenure),
            "MonthlyCharges": float(MonthlyCharges),
            "TotalCharges": float(TotalCharges),
            "gender": gender,
            "Partner": Partner,
            "Dependents": Dependents,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df)

        # Align with training features - fill missing columns with 0
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        # Scale numerical columns
        df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])

        # Make prediction
        pred = best_model.predict(df_encoded)[0]

        return "✅ Churn: Yes" if pred == 1 else "❎ Churn: No"

    except Exception as e:
        return f"❌ Error occurred: {str(e)}"

# Define Gradio inputs
inputs = [
    gr.Radio([0, 1], label="SeniorCitizen"),
    gr.Textbox(label="tenure"),
    gr.Textbox(label="MonthlyCharges"),
    gr.Textbox(label="TotalCharges"),
    gr.Dropdown(["Male", "Female"], label="gender"),
    gr.Dropdown(["Yes", "No"], label="Partner"),
    gr.Dropdown(["Yes", "No"], label="Dependents"),
    gr.Dropdown(["Yes", "No"], label="PhoneService"),
    gr.Dropdown(["Yes", "No", "No phone service"], label="MultipleLines"),
    gr.Dropdown(["DSL", "Fiber optic", "No"], label="InternetService"),
    gr.Dropdown(["Yes", "No", "No internet service"], label="OnlineSecurity"),
    gr.Dropdown(["Yes", "No", "No internet service"], label="OnlineBackup"),
    gr.Dropdown(["Yes", "No", "No internet service"], label="DeviceProtection"),
    gr.Dropdown(["Yes", "No", "No internet service"], label="TechSupport"),
    gr.Dropdown(["Yes", "No", "No internet service"], label="StreamingTV"),
    gr.Dropdown(["Yes", "No", "No internet service"], label="StreamingMovies"),
    gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
    gr.Dropdown(["Yes", "No"], label="PaperlessBilling"),
    gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="PaymentMethod")
]

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_churn,
    inputs=inputs,
    outputs="text",
    title="Customer Churn Predictor",
    description="Enter customer details to predict churn likelihood"
)

if __name__ == "__main__":
    interface.launch(share=True)