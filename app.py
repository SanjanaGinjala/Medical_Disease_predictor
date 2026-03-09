import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Medical Prediction", layout="centered")

st.title("🏥 Simple Medical Prediction App")
st.write("AI model to predict Diabetes Risk")

# Load model
model = joblib.load("models/diabetes_model.pkl")

# User inputs
pregnancies = st.slider("Pregnancies", 0, 10, 1)
glucose = st.slider("Glucose Level", 0, 200, 120)
bmi = st.slider("BMI", 10, 50, 25)
age = st.slider("Age", 20, 80, 30)

if st.button("Predict"):

    try:
        # prepare input
        data = np.array([[pregnancies, glucose, bmi, age]])

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("⚠ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

    except Exception as e:
        st.warning("Model input mismatch. Check training features.")
        st.write(str(e))