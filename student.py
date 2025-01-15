import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

st.write("""
# Student Performance Prediction App
This app predicts student exam scores and compares classification model performances!
""")

st.sidebar.header('Dataset Upload')
uploaded_file = st.sidebar.file_uploader("StudentPerformanceFactors.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader('Dataset Preview')
    st.write(df.head())

    categorical_columns = [
        'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
        'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    X = df.drop('Exam_Score', axis=1).to_numpy()
    y = df['Exam_Score'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=314)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader('Linear Regression Model Evaluation')
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-Squared (R2): {r2:.2f}")

    # Classification Models
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(gamma='auto')
    }

    accuracy_scores = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy_scores[name] = accuracy_score(y_test, y_pred)

    st.subheader('Classification Model Accuracies')
    accuracy_df = pd.DataFrame(list(accuracy_scores.items()), columns=["Model", "Accuracy"])
    st.write(accuracy_df)

else:
    st.write("StudentPerformanceFactors.csv")
