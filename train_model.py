"""
Trains the same Logistic Regression model used in the notebook
(final_nti_project_nti.ipynb) and saves it + the StandardScaler
so the Flask app can load them and make predictions.

Run this once before starting the Flask app:
    python train_model.py
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

FEATURES = [
    "Hours_Coding",
    "AI_Usage_Hours",
    "Lines_of_Code",
    "Commits",
    "Bugs_Reported",
    "Sleep_Hours",
    "Distractions",
    "Cognitive_Load",
    "Stress_Level",
]
TARGET = "Task_Success"

def main():
    df = pd.read_csv("ai_developer_productivity.csv")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(model, "logistic_regression_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and Scaler saved successfully!")

if __name__ == "__main__":
    main()
