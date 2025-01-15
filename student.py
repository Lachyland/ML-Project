import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the model (assuming the model is saved as 'studentP.pkl')
model = joblib.load('studentP.pkl')

# Load the dataset to get column info (not training here)
df = pd.read_csv("StudentPerformanceFactors.csv")

# Preprocessing
categorical_columns = [
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
    'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]

# Encode categorical features as the model expects them to be preprocessed (dummy variables)
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features and target (using only the necessary features)
X = df.drop('Exam_Score', axis=1).to_numpy()

# App setup
st.title("Student Performance Prediction")
st.write("""
    This app predicts a student's exam score based on various factors like hours studied, 
    attendance, parental involvement, and more. Fill in the following details to get a prediction.
""")

# Survey inputs for the user
hours_studied = st.number_input("Hours Studied (per week)", min_value=0, max_value=168, step=1)
attendance = st.slider("Attendance (%)", min_value=0, max_value=100, step=1)

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

# Process categorical inputs as the model expects them (like encoding)
inputs = pd.DataFrame([{
    'Hours_Studied': hours_studied,
    'Attendance': attendance,
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

# Encode categorical columns for prediction (like the training data)
for col in categorical_columns:
    inputs[col] = le.fit_transform(inputs[col])

# Prepare the input data for prediction (convert to numpy array)
X_input = inputs.to_numpy()

# Prediction
if st.button('Predict Exam Score'):
    prediction = model.predict(X_input)
    st.write(f"Predicted Exam Score: {prediction[0]:.2f}")
