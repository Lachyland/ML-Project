import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score

# Load the saved model
model = joblib.load('studentP.pkl')

# Load the dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

# Preprocessing
categorical_columns = [
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
    'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
    'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Features and target
X = df.drop('Exam_Score', axis=1).to_numpy()
y = df['Exam_Score'].to_numpy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=314)

# Linear Regression Model for baseline prediction
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Other Models
models = {
    "Logistic Regression": LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(gamma='auto')
}

accuracy_scores = {}

for name, clf in models.items():
    clf.fit(X_train, y_train)
    accuracy_scores[name] = accuracy_score(y_test, clf.predict(X_test))

# Streamlit Interface
st.write("""
# Student Performance Prediction App
This app predicts **student performance** and evaluates various machine learning models!
""")

st.subheader("Dataset Preview")
st.write(df.head())

# Sidebar for user input
st.sidebar.header("Input Features for Prediction")

# Collect user input for prediction
hours_studied = st.sidebar.slider("Hours Studied (per week)", 0, 60, 15)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 85)
parental_involvement = st.sidebar.selectbox("Parental Involvement", ["Low", "Medium", "High"])
access_to_resources = st.sidebar.selectbox("Access to Resources", ["Low", "Medium", "High"])
extracurricular_activities = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep_hours = st.sidebar.slider("Sleep Hours (per night)", 0, 12, 8)
previous_scores = st.sidebar.slider("Previous Scores", 0, 100, 70)
motivation_level = st.sidebar.selectbox("Motivation Level", ["Low", "Medium", "High"])
internet_access = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
tutoring_sessions = st.sidebar.slider("Tutoring Sessions (per month)", 0, 10, 2)
family_income = st.sidebar.selectbox("Family Income", ["Low", "Medium", "High"])
teacher_quality = st.sidebar.selectbox("Teacher Quality", ["Low", "Medium", "High"])
school_type = st.sidebar.selectbox("School Type", ["Public", "Private"])
peer_influence = st.sidebar.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])
physical_activity = st.sidebar.slider("Physical Activity (per week, hours)", 0, 20, 5)
learning_disabilities = st.sidebar.selectbox("Learning Disabilities", ["Yes", "No"])
parental_education_level = st.sidebar.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
distance_from_home = st.sidebar.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Prepare input data for prediction
input_data = {
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Parental_Involvement": parental_involvement,
    "Access_to_Resources": access_to_resources,
    "Extracurricular_Activities": extracurricular_activities,
    "Sleep_Hours": sleep_hours,
    "Previous_Scores": previous_scores,
    "Motivation_Level": motivation_level,
    "Internet_Access": internet_access,
    "Tutoring_Sessions": tutoring_sessions,
    "Family_Income": family_income,
    "Teacher_Quality": teacher_quality,
    "School_Type": school_type,
    "Peer_Influence": peer_influence,
    "Physical_Activity": physical_activity,
    "Learning_Disabilities": learning_disabilities,
    "Parental_Education_Level": parental_education_level,
    "Distance_from_Home": distance_from_home,
    "Gender": gender
}

# Convert categorical columns to numerical (dummy encoding)
input_df = pd.DataFrame([input_data])
input_df_encoded = pd.get_dummies(input_df)

# Ensure matching columns after encoding
input_df_encoded = input_df_encoded.reindex(columns=X_train.columns, fill_value=0)

# Make prediction using the loaded model
exam_score_prediction = model.predict(input_df_encoded)[0]

# Show the result
st.subheader("Prediction Result")
st.write(f"Predicted Exam Score: {exam_score_prediction:.2f}")

# Model Performance Metrics
st.subheader("Model Performance Comparison")
st.write(f"Linear Regression: MSE = {mse_lr:.2f}, R² = {r2_lr:.2f}")

# Create a dataframe for model accuracy scores
accuracy_df = pd.DataFrame(list(accuracy_scores.items()), columns=["Model", "Accuracy"])

# Display model comparison metrics
st.write(accuracy_df)
