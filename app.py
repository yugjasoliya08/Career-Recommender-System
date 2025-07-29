from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# Load model and encoders
model = load("model/model.pkl")
le_skills = load("model/skills_encoder.pkl")
le_interest = load("model/interest_encoder.pkl")
le_career = load("model/career_encoder.pkl")

def get_matching_label(user_input, class_list):
    for label in class_list:
        if label.lower() == user_input.lower():
            return label
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            gpa = float(request.form["gpa"])
            skill_input = request.form["skill"]
            interest_input = request.form["interest"]

            skill = get_matching_label(skill_input, le_skills.classes_)
            interest = get_matching_label(interest_input, le_interest.classes_)

            if skill is None or interest is None:
                error = "Skill or Interest not found in training data."
            else:
                skill_encoded = le_skills.transform([skill])[0]
                interest_encoded = le_interest.transform([interest])[0]

                pred = model.predict([[gpa, skill_encoded, interest_encoded]])
                prediction = le_career.inverse_transform(pred)[0]

        except Exception as e:
            error = "Invalid input! Please check your data."

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
