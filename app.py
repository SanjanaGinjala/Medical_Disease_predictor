import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Medical Disease Predictor")

st.title("🏥 Medical Disease Prediction System")

tab1, tab2, tab3 = st.tabs(["Diabetes", "Heart Disease", "Parkinsons"])


# ---------------- DIABETES ----------------
with tab1:

    st.header("Diabetes Prediction")

    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    data = pd.read_csv(url)

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    model = RandomForestClassifier()
    model.fit(X, y)

    preg = st.slider("Pregnancies",0,20,1)
    glucose = st.slider("Glucose",0,200,120)
    bp = st.slider("Blood Pressure",0,150,70)
    skin = st.slider("Skin Thickness",0,100,20)
    insulin = st.slider("Insulin",0,900,80)
    bmi = st.slider("BMI",10.0,50.0,25.0)
    dpf = st.slider("Diabetes Pedigree Function",0.0,2.5,0.5)
    age = st.slider("Age",20,80,30)

    if st.button("Predict Diabetes"):

        input_data = [[preg,glucose,bp,skin,insulin,bmi,dpf,age]]

        result = model.predict(input_data)

        if result[0] == 1:
            st.error("⚠ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk")


# ---------------- HEART ----------------
with tab2:

    st.header("Heart Disease Prediction")

    url = "https://raw.githubusercontent.com/anishathalye/neural-style/master/examples/heart.csv"
    data = pd.read_csv(url)

    X = data.drop("target", axis=1)
    y = data["target"]

    model = RandomForestClassifier()
    model.fit(X, y)

    age = st.slider("Age",20,80,40)
    sex = st.selectbox("Sex",[0,1])
    cp = st.selectbox("Chest Pain Type",[0,1,2,3])
    trestbps = st.slider("Resting BP",90,200,120)
    chol = st.slider("Cholesterol",100,400,200)
    thalach = st.slider("Max Heart Rate",70,210,150)

    if st.button("Predict Heart Disease"):

        input_data = [[age,sex,cp,trestbps,chol,0,1,thalach,0,1,1,0,2]]

        result = model.predict(input_data)

        if result[0] == 1:
            st.error("⚠ Heart Disease Risk")
        else:
            st.success("✅ Healthy")


# ---------------- PARKINSONS ----------------
with tab3:

    st.header("Parkinsons Prediction")

    url = "https://raw.githubusercontent.com/plotly/datasets/master/parkinsons.csv"
    data = pd.read_csv(url)

    X = data.drop(["name","status"], axis=1)
    y = data["status"]

    model = RandomForestClassifier()
    model.fit(X, y)

    fo = st.slider("Voice Frequency (Fo)",80,260,120)

    if st.button("Predict Parkinsons"):

        input_data = [[fo]*22]

        result = model.predict(input_data)

        if result[0] == 1:
            st.error("⚠ Parkinsons Detected")
        else:
            st.success("✅ Normal")