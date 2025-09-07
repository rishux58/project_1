import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Sidebar / Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Predict"],
        icons=["house", "activity"],  # icons from bootstrap
        menu_icon="cast",
        default_index=0,
    )

# ================= HOME PAGE =================
import streamlit as st
import streamlit.components.v1 as components

def home_page():
    components.html(
        """
        <div style="
            background-color: #262730;
            padding: 25px;
            border-radius: 15px;
            margin-top: 20px;
        ">
            <h1 style="text-align: center; color: #FF4B4B;">🩺 Disease Prediction System</h1>

            <h2 style="color: #FFFFFF; text-align: center;">Welcome to the Diabetes Prediction App ⚡</h2>

            <p style="color: #BBBBBB; font-size: 16px; text-align:center;">
                This application helps you predict whether a patient is 
                <b style="color:#FF4B4B;">Diabetic</b> or 
                <b style="color:#4CAF50;">Healthy</b> 
                based on their medical details like glucose level, BMI, age, insulin, etc.
            </p>

            <p style="color: #DDDDDD; font-size: 17px; line-height: 1.8;">
                ✅ Simple & User Friendly <br>
                ⚡ Fast & Accurate Predictions <br>
                🤖 Powered by Machine Learning
            </p>

            <p style="text-align:center; color:#AAAAAA; font-size: 15px; margin-top: 15px;">
                👉 Use the sidebar to switch to the <b style="color:#FF4B4B;">Predict</b> page 🚀
            </p>
        </div>
        """,
        height=400,
    )



# ================= PREDICTION PAGE =================
def predict_page():
    st.title("🩸 Diabetes Prediction")
    st.write("Fill the form below to check health status.")

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

    # Form inputs
    preg = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 200, 100)
    bp = st.number_input("Blood Pressure", 0, 122, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

    if st.button("Predict"):
        new_patient = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        new_patient = scaler.transform(new_patient)
        prediction = model.predict(new_patient)

        if prediction[0] == 1:
            st.error("Prediction: Diabetic 🩸")
        else:
            st.success("Prediction: Healthy ✅")

# ================= MAIN APP =================
if selected == "Home":
    home_page()
elif selected == "Predict":
    predict_page()
