from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler (created by train_model.py)
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Must match the exact order used during training
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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            values = [float(request.form[feat]) for feat in FEATURES]
            input_array = np.array(values).reshape(1, -1)
            scaled_input = scaler.transform(input_array)

            pred = model.predict(scaled_input)[0]
            proba = model.predict_proba(scaled_input)[0][1]  # P(Task_Success = 1)

            prediction = "Success" if pred == 1 else "Not Successful"
            probability = round(proba * 100, 2)
        except (ValueError, KeyError):
            error = "Please fill in all fields with valid numbers."

    return render_template(
        "index.html",
        features=FEATURES,
        prediction=prediction,
        probability=probability,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)
