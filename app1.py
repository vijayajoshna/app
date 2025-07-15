import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return round(weight / (height_m ** 2), 2)

def element_based_prediction():
    st.subheader("🧪 Health Input Method")

    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0)
    age = st.number_input("Age", min_value=10, max_value=100)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=50.0, max_value=300.0)
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=50.0, max_value=200.0)
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20)
    insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0)

    if st.button("Predict (with health data)"):
        bmi = calculate_bmi(weight, height)

        X_input = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, 0.0, age, 0.0, 0.0]])
        X_scaled = scaler.transform(X_input)

        prob = model.predict_proba(X_scaled)[0][1] * 100
        prediction = "Likely Diabetic" if prob >= 60 else ("Borderline Risk" if prob >= 30 else "Non-Diabetic")

        st.success(f"Prediction: {prediction} ({prob:.2f}%)")
        st.info(f"Calculated BMI: {bmi}")

def symptom_based_prediction():
    st.subheader("🔍 Symptom Checker")

    q1 = st.checkbox("Frequent urination")
    q2 = st.checkbox("Excessive thirst")
    q3 = st.checkbox("Unexplained weight loss")
    q4 = st.checkbox("Blurred vision")
    q5 = st.checkbox("Fatigue or weakness")
    q6 = st.checkbox("Slow healing wounds")
    q7 = st.checkbox("Tingling in hands/feet")

    symptoms = [q1, q2, q3, q4, q5, q6, q7]
    score = sum(symptoms)

    if st.button("Check Symptoms"):
        if score >= 5:
            st.error("⚠️ High Risk of Diabetes")
        elif score >= 3:
            st.warning("⚠️ Moderate Risk — Please consult a doctor")
        else:
            st.success("✅ Low Risk based on symptoms")

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")

tab1, tab2 = st.tabs(["📊 Element-Based", "🧠 Symptom-Based"])

with tab1:
    element_based_prediction()

with tab2:
    symptom_based_prediction()
