import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
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

# Collect user input
motivation_level = st.sidebar.slider("Motivation Level", 1, 10, 5)
family_income = st.sidebar.slider("Family Income", 1000, 50000, 25000)
internet_access = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Encode categorical inputs
input_data = {
    "Motivation_Level": motivation_level,
    "Family_Income": family_income,
    "Internet_Access": 1 if internet_access == "Yes" else 0,
    "Gender_Female": 1 if gender == "Female" else 0,
}

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

# Plot the accuracy comparison
fig, ax = plt.subplots()
sns.barplot(x="Accuracy", y="Model", data=accuracy_df, ax=ax, palette="viridis")
st.pyplot(fig)

# Display model comparison metrics
st.write(accuracy_df)

# Show the model's MSE and R² for Linear Regression
st.write("""
### Performance Summary for Linear Regression:
- **Mean Squared Error (MSE)**: The MSE measures the average squared difference between the predicted and actual values. Lower values are better.
- **R-Squared (R²)**: R² measures how well the model's predictions match the actual data. Higher values indicate better performance.
""")
