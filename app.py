import streamlit as st
from predict import predict_heart, predict_diabetes, predict_parkinsons

st.set_page_config(
    page_title="Medical Disease Prediction",
    layout="wide"
)

st.title("🏥 Medical Disease Prediction System")
st.write("AI based system to predict major diseases")

tab1, tab2, tab3 = st.tabs([
    "❤️ Heart Disease",
    "🩸 Diabetes",
    "🧠 Parkinsons"
])

# HEART DISEASE TAB
with tab1:

    st.header("Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age",20,100,40)
        sex = st.selectbox("Sex",[0,1])
        cp = st.selectbox("Chest Pain Type",[0,1,2,3])
        trestbps = st.slider("Resting Blood Pressure",90,200,120)

    with col2:
        chol = st.slider("Cholesterol",100,400,200)
        thalach = st.slider("Max Heart Rate",70,210,150)
        exang = st.selectbox("Exercise Induced Angina",[0,1])
        oldpeak = st.slider("Old Peak",0.0,6.0,1.0)

    if st.button("Predict Heart Disease"):

        result = predict_heart([
            age,sex,cp,trestbps,chol,
            0,1,thalach,exang,
            oldpeak,1,0,2
        ])

        if result[0]==1:
            st.error("⚠ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk")


# DIABETES TAB
with tab2:

    st.header("Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Pregnancies",0,10,1)
        glucose = st.slider("Glucose Level",0,200,120)

    with col2:
        bmi = st.slider("BMI",10,50,25)
        age = st.slider("Age",20,80,30)

    if st.button("Predict Diabetes"):

        result = predict_diabetes([
            pregnancies,glucose,70,20,80,bmi,0.5,age
        ])

        if result[0]==1:
            st.error("⚠ Diabetes Detected")
        else:
            st.success("✅ No Diabetes")


# PARKINSONS TAB
with tab3:

    st.header("Parkinsons Prediction")

    fo = st.slider("MDVP Fo Frequency",80,260,120)

    if st.button("Predict Parkinsons"):

        result = predict_parkinsons([fo]*22)

        if result[0]==1:
            st.error("⚠ Parkinsons Detected")
        else:
            st.success("✅ No Parkinsons")