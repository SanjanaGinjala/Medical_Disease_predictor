import streamlit as st
from predict import predict_heart, predict_diabetes, predict_parkinsons

st.set_page_config(page_title="Medical Disease Prediction", layout="wide")

st.title("🏥 Medical Disease Prediction System")
st.write("AI based system to predict major diseases")

tab1, tab2, tab3 = st.tabs([
    "❤️ Heart Disease",
    "🩸 Diabetes",
    "🧠 Parkinsons"
])

# ================= HEART DISEASE =================
with tab1:

    st.header("Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age",20,100,40)
        sex = st.selectbox("Sex",[0,1])
        cp = st.selectbox("Chest Pain Type",[0,1,2,3])
        trestbps = st.slider("Resting Blood Pressure",90,200,120)
        chol = st.slider("Cholesterol",100,400,200)
        fbs = st.selectbox("Fasting Blood Sugar >120",[0,1])
        restecg = st.selectbox("Rest ECG",[0,1,2])

    with col2:
        thalach = st.slider("Max Heart Rate",70,210,150)
        exang = st.selectbox("Exercise Induced Angina",[0,1])
        oldpeak = st.slider("Old Peak",0.0,6.0,1.0)
        slope = st.selectbox("Slope",[0,1,2])
        ca = st.slider("Major Vessels (0-3)",0,3,0)
        thal = st.selectbox("Thal",[0,1,2,3])

    if st.button("Predict Heart Disease"):

        result = predict_heart([
            age,sex,cp,trestbps,chol,fbs,restecg,
            thalach,exang,oldpeak,slope,ca,thal
        ])

        if result[0]==1:
            st.error("⚠ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk")


# ================= DIABETES =================
with tab2:

    st.header("Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Pregnancies",0,10,1)
        glucose = st.slider("Glucose Level",0,200,120)
        blood_pressure = st.slider("Blood Pressure",40,140,70)
        skin_thickness = st.slider("Skin Thickness",0,100,20)

    with col2:
        insulin = st.slider("Insulin Level",0,300,80)
        bmi = st.slider("BMI",10,50,25)
        pedigree = st.slider("Diabetes Pedigree Function",0.0,2.5,0.5)
        age = st.slider("Age",20,80,30)

    if st.button("Predict Diabetes"):

        result = predict_diabetes([
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, pedigree, age
        ])

        if result[0]==1:
            st.error("⚠ Diabetes Detected")
        else:
            st.success("✅ No Diabetes")


# ================= PARKINSONS =================
with tab3:

    st.header("Parkinsons Prediction")

    st.write("Enter voice frequency values")

    fo = st.slider("MDVP Fo Frequency",80,260,120)

    if st.button("Predict Parkinsons"):

        # model expects 22 features
        result = predict_parkinsons([fo]*22)

        if result[0]==1:
            st.error("⚠ Parkinsons Detected")
        else:
            st.success("✅ No Parkinsons")