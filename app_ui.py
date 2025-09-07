import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict if they are **Diabetic** or **Healthy**.")

# Input fields
preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 200, 100)
bp = st.number_input("Blood Pressure", 0, 122, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

# Prediction button
if st.button("Predict"):
    new_patient = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    new_patient = scaler.transform(new_patient)
    prediction = model.predict(new_patient)

    if prediction[0] == 1:
        st.error("Prediction: Diabetic 🩸")
    else:
        st.success("Prediction: Healthy ✅")
