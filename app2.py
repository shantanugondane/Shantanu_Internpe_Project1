import streamlit as st
import numpy as np
import pandas as pd

import pickle
from streamlit_option_menu import option_menu


data = pd.read_csv(
    "D:/face mask detector/January-AI-ML-Internship/Internship Kartik/diabetes (1).csv"
)

loaded_model = pickle.load(open("Diabetesmodel.pkl", "rb"))

selected = option_menu(
    menu_title=None,
    options=["Predict Diabetes", "About", "Contribute to Dataset"],
    icons=["search", "search", "book"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)


if selected == "Predict Diabetes":
    st.title("DIABETES PREDICTION SYSTEM")
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=100, step=1)
    Glucose = st.number_input("Glucose", min_value=0, max_value=200, step=1)

    BloodPressure = st.number_input("BloodPressure", min_value=0, max_value=100, step=1)
    SkinThickness = st.number_input(
        "Skin Thickness", min_value=0, max_value=100, step=1
    )

    Insulin = st.number_input("Insulin", min_value=0, max_value=100, step=1)
    bmi = st.number_input(
        "Body Mass Index (BMI)", min_value=0.0, max_value=100.0, step=0.1
    )

    DPF = st.number_input(
        "Diabetes Predigree Function",
        min_value=0.000,
        max_value=100.000,
        step=0.001,
        format="%0.3f",
    )
    age = st.number_input("Age", min_value=0, max_value=100, step=1)

    pred = loaded_model.predict(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, bmi, DPF, age]]
    )

    if st.button("Predict"):
        st.write(pred)
        if pred[0] == 1:
            st.write("The person is diabetic")
        if pred[0] == 0:
            st.write("The person is not diabetic")


if selected == "About":
    st.header("What is Diabetes Prediction System?")

    st.write(
        "A diabetes prediction system is a computational model or tool designed to predict the likelihood of an individual developing diabetes in the future. It typically utilizes various data inputs, such as demographic information, medical history, lifestyle factors, and possibly genetic data, to assess the risk of diabetes onset. The goal of such systems is to identify individuals at high risk early on, allowing for preventive measures, lifestyle changes, or medical interventions to reduce the risk or manage the condition effectively.These prediction systems often leverage machine learning algorithms and statistical models to analyze large datasets and identify patterns or correlations that may be indicative of diabetes risk. The development of these systems may involve training the models on historical data from individuals with and without diabetes to learn patterns and associations."
    )

    st.header("Key Components of this System")
    st.write(
        "Data Collection: Gathering relevant data such as age, gender, family history, blood glucose levels, body mass index (BMI), physical activity, diet, and other health-related information."
    )

    st.write(
        "Feature Selection: Identifying the most significant factors or features that contribute to diabetes risk."
    )

    st.write(
        "Model Development: Building machine learning models or statistical algorithms that can predict the probability of an individual developing diabetes based on the collected data."
    )

    st.write(
        "Validation: Testing the accuracy and reliability of the model using independent datasets to ensure that it generalizes well to new, unseen data."
    )

    st.write(
        "Deployment: Implementing the prediction system in real-world settings, such as healthcare institutions or digital platforms, to assist healthcare professionals in assessing diabetes risk."
    )


if selected == "Contribute to Dataset":
    st.header("Contribute to our dataset")
    preg = st.number_input("Enter Pregnancies", 0, 20)
    gluc = st.number_input("Enter glucose", 0, 200, step=20)
    bp = st.number_input("Enter Blood Pressure", 0, 200)
    skinth = st.number_input("Enter Skin Thickness", 0.00, 100.00, step=10.0)
    ins = st.number_input("Enter Insulin", 0, 20)
    BMI = st.number_input("Enter BMI", 0.00, 100.00, step=10.0)
    dpf = st.number_input("Enter Diabetes Pedigree Function", 0.00, 100.00)
    Age = st.number_input("Enter Age", 0, 100, step=1)
    out = st.number_input("Output (0 or 1)", 0, 1)

    if st.button("Submit"):
        to_add = {
            "Pregnancies": [preg],
            "Glucose": [gluc],
            "Blood Pressure": [bp],
            "Skin Thickness": [skinth],
            "Insulin": [ins],
            "BMI": [BMI],
            "DiabetesPedigreeFunction": [dpf],
            "Age": [Age],
            "Outcome": [out],
        }
        to_add = pd.DataFrame(to_add)
        to_add.to_csv(
            "D:/face mask detector/January-AI-ML-Internship/Internship Kartik/diabetes (1).csv",
            mode="a",
            header=False,
            index=False,
        )
        st.success("Submitted")
    if st.checkbox("Show Table"):
        st.table(data)
