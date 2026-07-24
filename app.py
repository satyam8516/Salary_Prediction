import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Salary Prediction", page_icon="💰")

st.title("💰 Employee Salary Prediction")
st.write("Enter employee details to predict Annual Salary (LPA).")

# Input Fields
age = st.number_input("Age", 18, 65, 30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

education = st.selectbox(
    "Education",
    ["High School", "Bachelor", "Master", "PhD"]
)

experience = st.number_input("Experience (Years)", 0, 40, 5)

department = st.text_input("Department")

job_level = st.number_input("Job Level", 1, 10, 2)

performance = st.slider("Performance Rating", 1, 5, 3)

certifications = st.number_input("Certifications", 0, 20, 2)

overtime = st.number_input("Overtime Hours", 0, 100, 10)

remote = st.selectbox(
    "Remote Work",
    ["No", "Yes"]
)

city = st.text_input("City")

company_tenure = st.number_input("Company Tenure", 0, 40, 5)

projects = st.number_input("Projects Completed", 0, 100, 5)

skill = st.slider("Skill Score", 0, 100, 80)

# Prediction
if st.button("Predict Salary"):

    gender = encoders["Gender"].transform([gender])[0]
    education = encoders["Education"].transform([education])[0]
    department = encoders["Department"].transform([department])[0]
    job_level = encoders["Job_Level"].transform([job_level])[0]
    remote = encoders["Remote_Work"].transform([remote])[0]
    city = encoders["City"].transform([city])[0]

    df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education": education,
        "Experience_Years": experience,
        "Department": department,
        "Job_Level": job_level,
        "Performance_Rating": performance,
        "Certifications": certifications,
        "Overtime_Hours": overtime,
        "Remote_Work": remote,
        "City": city,
        "Company_Tenure": company_tenure,
        "Projects_Completed": projects,
        "Skill_Score": skill
    }])

    scaled = scaler.transform(df)
    prediction = model.predict(scaled)

    st.success(f"💰 Predicted Annual Salary: {prediction[0]:.2f} LPA")
