import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load('studentP.pkl')

# Function to preprocess input data
def preprocess_data(input_data):
    # Handle categorical variables (encode them)
    categorical_columns = [
        'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
        'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
    
    # Label encoding for categorical columns
    for column in categorical_columns:
        le = LabelEncoder()
        input_data[column] = le.fit_transform(input_data[column])
    
    # Convert the input data to a numpy array and return it
    return input_data.to_numpy()

# Function to get the input data from the user
def get_user_input():
    # Get input from the user
    st.title('Student Exam Score Predictor')
    
    hours_studied = st.slider('Hours Studied', 0, 168, 10)
    attendance = st.slider('Attendance Percentage', 0, 100, 85)
    parental_involvement = st.selectbox('Parental Involvement', ['Low', 'Medium', 'High'])
    access_to_resources = st.selectbox('Access to Resources', ['Low', 'Medium', 'High'])
    extracurricular_activities = st.selectbox('Extracurricular Activities', ['Yes', 'No'])
    sleep_hours = st.slider('Sleep Hours', 0, 24, 7)
    previous_scores = st.slider('Previous Exam Scores', 0, 100, 60)
    motivation_level = st.selectbox('Motivation Level', ['Low', 'Medium', 'High'])
    internet_access = st.selectbox('Internet Access', ['Yes', 'No'])
    tutoring_sessions = st.slider('Tutoring Sessions per Month', 0, 10, 2)
    family_income = st.selectbox('Family Income', ['Low', 'Medium', 'High'])
    teacher_quality = st.selectbox('Teacher Quality', ['Low', 'Medium', 'High'])
    school_type = st.selectbox('School Type', ['Public', 'Private'])
    peer_influence = st.selectbox('Peer Influence', ['Positive', 'Neutral', 'Negative'])
    physical_activity = st.slider('Physical Activity Hours per Week', 0, 20, 3)
    learning_disabilities = st.selectbox('Learning Disabilities', ['Yes', 'No'])
    parental_education_level = st.selectbox('Parental Education Level', ['High School', 'College', 'Postgraduate'])
    distance_from_home = st.selectbox('Distance from Home', ['Near', 'Moderate', 'Far'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    
    # Store the input data in a dictionary
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
    
    # Convert the input dictionary into a dataframe
    input_df = pd.DataFrame([input_data])
    
    return input_df

# Make predictions using the trained model
def predict(input_df):
    processed_input = preprocess_data(input_df)
    prediction = model.predict(processed_input)
    return prediction[0]

# Display the user input
input_df = get_user_input()
st.write("Input data:", input_df)

# Make predictions
if st.button('Predict Exam Score'):
    prediction = predict(input_df)
    st.write(f"Predicted Exam Score: {prediction:.2f}")
