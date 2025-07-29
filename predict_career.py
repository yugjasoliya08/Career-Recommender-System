import pandas as pd
from joblib import load

# Load trained model and encoders
model = load("model/model.pkl")
le_skills = load("model/skills_encoder.pkl")
le_interest = load("model/interest_encoder.pkl")
le_career = load("model/career_encoder.pkl")

# ----------- Helper Function ------------
def get_matching_label(user_input, class_list):
    for label in class_list:
        if label.lower() == user_input.lower():
            return label
    return None

# ----------- Input Section ------------
print("\nWelcome to Career Recommender System 💼\n")

try:
    gpa = float(input("Enter your GPA (0.0 to 10.0): "))
    skill_input = input("Enter your Skill (e.g., Python, Design, Analysis): ").strip()
    interest_input = input("Enter your Interest (e.g., Programming, Art, Data): ").strip()

    # Validate skill and interest
    skill = get_matching_label(skill_input, le_skills.classes_)
    interest = get_matching_label(interest_input, le_interest.classes_)

    if skill is None or interest is None:
        print(f"\n⚠️ Error: Entered Skill or Interest not found in training data.")
        print(f"Available Skills: {list(le_skills.classes_)}")
        print(f"Available Interests: {list(le_interest.classes_)}")
        exit()

    # Encode validated input
    skill_encoded = le_skills.transform([skill])[0]
    interest_encoded = le_interest.transform([interest])[0]

    # Predict career
    prediction = model.predict([[gpa, skill_encoded, interest_encoded]])
    recommended_career = le_career.inverse_transform(prediction)[0]

    print(f"\n✅ Based on your input, Recommended Career: **{recommended_career}**")

except Exception as e:
    print("\n⚠️ Error:", e)
    print("Please enter valid data.")
