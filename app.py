import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🎓 Student Performance Predictor")

st.write("Enter student details to predict Math Score")

gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

race = st.selectbox(
    "Race/Ethnicity",
    [
        "group A",
        "group B",
        "group C",
        "group D",
        "group E"
    ]
)

parental_education = st.selectbox(
    "Parental Level of Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)

lunch = st.selectbox(
    "Lunch Type",
    [
        "standard",
        "free/reduced"
    ]
)

prep = st.selectbox(
    "Test Preparation Course",
    [
        "none",
        "completed"
    ]
)

reading_score = st.slider(
    "Reading Score",
    0,
    100,
    50
)

writing_score = st.slider(
    "Writing Score",
    0,
    100,
    50
)

if st.button("Predict Math Score"):

    input_data = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [parental_education],
        "lunch": [lunch],
        "test preparation course": [prep],
        "reading score": [reading_score],
        "writing score": [writing_score]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Math Score: {prediction[0]:.2f}"
    )