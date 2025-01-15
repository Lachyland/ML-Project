import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('studentP.pkl')

# Set the path to your dataset or CSV
file_path = 'student_data.csv'  # Adjust the path to your data file if needed

st.write("""
# Student Performance Prediction App
This app predicts **student performance** based on input data!
""")

# Survey form to get user input
st.subheader("Survey: Please fill in the details")

# Input fields
hours_studied = st.number_input("Hours Studied (per week)", min_value=0, max_value=168, value=10)
attendance = st.slider("Attendance Percentage", min_value=0, max_value=100, value=85)
parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep_hours = st.number_input("Average Sleep Hours (per night)", min_value=0, max_value=24, value=7)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=65)
motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
internet_access = st.selectbox("Internet Access", ["Yes", "No"])
tutoring_sessions = st.number_input("Tutoring Sessions (per month)", min_value=0, max_value=100, value=2)
family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
school_type = st.selectbox("School Type", ["Public", "Private"])
peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])
physical_activity = st.number_input("Physical Activity (hours per week)", min_value=0, max_value=168, value=5)
learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
parental_education_level = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Convert categorical data to the appropriate format
input_data = {
    'Hours_Studied': hours_studied,
    'Attendance': attendance,
    'Parental_Involvement': parental_involvement,
    'Access_to_Resources': access_to_resources,
    'Extracurricular_Activities': extracurricular_activities,
    'Sleep_Hours': sleep_hours,
    'Previous_Scores': previous_scores,
    'Motivation_Level': motivation_level,
    'Internet_Access': internet_access,
    'Tutoring_Sessions': tutoring_sessions,
    'Family_Income': family_income,
    'Teacher_Quality': teacher_quality,
    'School_Type': school_type,
    'Peer_Influence': peer_influence,
    'Physical_Activity': physical_activity,
    'Learning_Disabilities': learning_disabilities,
    'Parental_Education_Level': parental_education_level,
    'Distance_from_Home': distance_from_home,
    'Gender': gender
}

# Convert categorical data into one-hot encoded format
input_df = pd.DataFrame([input_data])

# Handle one-hot encoding for categorical columns based on your original model training
# Assuming 'get_dummies' was used for encoding in the training model
categorical_columns = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                       'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                       'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level',
                       'Distance_from_Home', 'Gender']

input_df = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

# Ensure the input matches the columns of the trained model
# You might need to add missing columns if necessary
missing_cols = set(X.columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0  # Add the missing columns with default value 0

input_df = input_df[X.columns]  # Align the input dataframe with the model features

# Make a prediction
prediction = model.predict(input_df)[0]

# Show the prediction
st.subheader(f"Predicted Exam Score: {prediction:.2f}")
