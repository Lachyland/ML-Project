import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the model
model = joblib.load('studentP.pkl')

# Load the dataset to get column info (not training here)
df = pd.read_csv("StudentPerformanceFactors.csv")

# Preprocessing - ensure this matches the training process
categorical_columns = [
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
    'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

# Encode categorical fields to match training
le_dict = {}
for col in categorical_columns:
    le_dict[col] = LabelEncoder()
    df[col] = le_dict[col].fit_transform(df[col])

# Features (X) for training the model
X = df.drop('Exam_Score', axis=1)

# App setup
st.title("Student Performance Prediction")
st.write("""
    This app predicts a student's exam score based on various factors like hours studied, 
    attendance, parental involvement, and more. Fill in the following details to get a prediction.
""")

# Survey inputs for the user
hours_studied = st.number_input("Hours Studied (per week)", min_value=0, max_value=168, step=1)
attendance = st.slider("Attendance (%)", min_value=0, max_value=100, step=1)
sleep_hours = st.number_input("Sleep Hours (per night)", min_value=0, max_value=24, step=1)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, step=1)
tutoring_sessions = st.number_input("Tutoring Sessions (per month)", min_value=0, max_value=31, step=1)
physical_activity = st.number_input("Physical Activity (hours per week)", min_value=0, max_value=168, step=1)

# Categorical fields for the user to select
parental_involvement = st.selectbox('Parental Involvement', ['Low', 'Medium', 'High'])
access_to_resources = st.selectbox('Access to Resources', ['Low', 'Medium', 'High'])
extracurricular_activities = st.selectbox('Extracurricular Activities', ['Yes', 'No'])
motivation_level = st.selectbox('Motivation Level', ['Low', 'Medium', 'High'])
internet_access = st.selectbox('Internet Access', ['Yes', 'No'])
family_income = st.selectbox('Family Income', ['Low', 'Medium', 'High'])
teacher_quality = st.selectbox('Teacher Quality', ['Low', 'Medium', 'High'])
school_type = st.selectbox('School Type', ['Public', 'Private'])
peer_influence = st.selectbox('Peer Influence', ['Positive', 'Neutral', 'Negative'])
learning_disabilities = st.selectbox('Learning Disabilities', ['Yes', 'No'])
parental_education_level = st.selectbox('Parental Education Level', ['High School', 'College', 'Postgraduate'])
distance_from_home = st.selectbox('Distance from Home', ['Near', 'Moderate', 'Far'])
gender = st.selectbox('Gender', ['Male', 'Female'])

# Create the input dataframe for prediction
inputs = pd.DataFrame([{
    'Hours_Studied': hours_studied,
    'Attendance': attendance,
    'Sleep_Hours': sleep_hours,
    'Previous_Scores': previous_scores,
    'Tutoring_Sessions': tutoring_sessions,
    'Physical_Activity': physical_activity,
    'Parental_Involvement': parental_involvement,
    'Access_to_Resources': access_to_resources,
    'Extracurricular_Activities': extracurricular_activities,
    'Motivation_Level': motivation_level,
    'Internet_Access': internet_access,
    'Family_Income': family_income,
    'Teacher_Quality': teacher_quality,
    'School_Type': school_type,
    'Peer_Influence': peer_influence,
    'Learning_Disabilities': learning_disabilities,
    'Parental_Education_Level': parental_education_level,
    'Distance_from_Home': distance_from_home,
    'Gender': gender
}])

# Encode categorical columns for prediction to match training data
for col in categorical_columns:
    inputs[col] = le_dict[col].transform(inputs[col])

# Align input dataframe columns with the model's expected features
input_df = inputs[X.columns]  # Ensure the same columns as the training data

# Make a prediction
if st.button('Predict Exam Score'):
    prediction = model.predict(input_df)[0]
    st.subheader(f"Predicted Exam Score: {prediction:.2f}")