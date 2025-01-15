import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Set the path to your local CSV file
file_path = 'StudentPerformanceFactors.csv'

st.write("""
# Student Performance Prediction App
This app predicts **student performance** and evaluates various machine learning models!
""")

# Load the dataset directly from the file path
df = pd.read_csv(file_path)

st.subheader("Dataset Preview")
st.write(df.head())

# Preprocessing
st.sidebar.header("Categorical Columns")
categorical_columns = st.sidebar.multiselect(
    "Select categorical columns",
    options=df.columns,
    default=[
        'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Learning_Disabilities',
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
)

if categorical_columns:
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Features and target
X = df.drop('Exam_Score', axis=1).to_numpy()
y = df['Exam_Score'].to_numpy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=314)

# Linear Regression
st.subheader("Linear Regression")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-Squared (R2): {r2:.2f}")

# Other Models
st.subheader("Other Models Accuracy")
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

accuracy_df = pd.DataFrame(list(accuracy_scores.items()), columns=["Model", "Accuracy"])
st.write(accuracy_df)