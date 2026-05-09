import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# Sample Customer Churn Dataset


data = {
    "Tenure": [1, 2, 3, 5, 7, 10, 12, 15, 18, 20],
    "MonthlyCharges": [20, 30, 40, 50, 70, 90, 100, 120, 150, 170],
    "SupportCalls": [5, 4, 4, 3, 2, 1, 0, 0, 0, 0],
    "Churn": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)



# App Title


st.title("Customer Churn Prediction App")

st.write(
    "This app predicts whether a customer is likely to leave the company."
)



# Show Dataset


st.subheader("Dataset")

st.dataframe(df)



# Prepare Data


X = df[["Tenure", "MonthlyCharges", "SupportCalls"]]
y = df["Churn"]



# Split Data


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# Train Model


model = LogisticRegression()

model.fit(X_train, y_train)



# Model Accuracy


prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

st.subheader("Model Accuracy")

st.write(f"Accuracy: {accuracy:.2f}")



# User Input Section


st.subheader("Enter Customer Details")

tenure = st.number_input(
    "Years with Company",
    min_value=0,
    max_value=50,
    value=5
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0,
    max_value=1000,
    value=50
)

support_calls = st.number_input(
    "Number of Support Calls",
    min_value=0,
    max_value=20,
    value=2
)



# Predict Churn


if st.button("Predict"):

    result = model.predict(
        pd.DataFrame(
            [[tenure, monthly_charges, support_calls]],
            columns=["Tenure", "MonthlyCharges", "SupportCalls"]
        )
    )

    if result[0] == 1:
        st.error("Customer is likely to leave the company.")
    else:
        st.success("Customer is likely to stay.")