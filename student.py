import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model
model = joblib.load('studentP.pkl')

# Define function to process the input data
def process_data(data):
    # Define categorical columns
    categorical_columns = [
        'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Learning_Disabilities',
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
    
    # Convert input to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Convert categorical columns to dummy variables
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Ensure the DataFrame has the same columns as the model
    try:
        model_columns = model.feature_names_in_
    except AttributeError:
        raise AttributeError("The model is missing the 'feature_names_in_' attribute.")
    
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with zero values
    
    # Reorder columns to match the training data structure
    df = df[model_columns]
    return df


# Streamlit User Interface
st.title("Student Exam Score Prediction")

st.sidebar.header("Enter Student Data")

# Inputs
hours_studied = st.sidebar.number_input("Hours Studied", min_value=0, max_value=168, value=10)
attendance = st.sidebar.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
parental_involvement = st.sidebar.selectbox("Parental Involvement", ["Low", "Medium", "High"])
access_to_resources = st.sidebar.selectbox("Access to Resources", ["Low", "Medium", "High"])
extracurricular_activities = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep_hours = st.sidebar.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
previous_scores = st.sidebar.number_input("Previous Scores", min_value=0, max_value=100, value=60)
motivation_level = st.sidebar.selectbox("Motivation Level", ["Low", "Medium", "High"])
internet_access = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
tutoring_sessions = st.sidebar.number_input("Tutoring Sessions (per month)", min_value=0, max_value=30, value=2)
family_income = st.sidebar.selectbox("Family Income", ["Low", "Medium", "High"])
teacher_quality = st.sidebar.selectbox("Teacher Quality", ["Low", "Medium", "High"])
school_type = st.sidebar.selectbox("School Type", ["Public", "Private"])
peer_influence = st.sidebar.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])
physical_activity = st.sidebar.number_input("Physical Activity (hrs/week)", min_value=0, max_value=168, value=5)
learning_disabilities = st.sidebar.selectbox("Learning Disabilities", ["Yes", "No"])
parental_education_level = st.sidebar.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
distance_from_home = st.sidebar.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Button for prediction
if st.sidebar.button("Predict Exam Score"):
    # Collect input data
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
    
    # Process the data and make predictions
    processed_data = process_data(input_data)
    predicted_score = model.predict(processed_data)
    
    # Show the predicted exam score
    st.subheader(f"Predicted Exam Score: {predicted_score[0]:.2f}")
