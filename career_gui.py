import streamlit as st
import pandas as pd
from joblib import load

# Load trained model and encoders
model = load("model/model.pkl")
le_skills = load("model/skills_encoder.pkl")
le_interest = load("model/interest_encoder.pkl")
le_career = load("model/career_encoder.pkl")

# Helper function to get correct label
def get_matching_label(user_input, class_list):
    for label in class_list:
        if label.lower() == user_input.lower():
            return label
    return None

# ---------------- UI Design ----------------
st.set_page_config(page_title="Career Recommender 💼", page_icon="💼")
st.title("🎓 Career Recommender System")
st.markdown("Enter your details below to get a personalized career recommendation:")

st.markdown("---")

# ---------------- Input Section ----------------
gpa = st.slider("Select your GPA:", min_value=0.0, max_value=10.0, step=0.1)

skill = st.selectbox("Select your Skill:", sorted(le_skills.classes_))
interest = st.selectbox("Select your Interest:", sorted(le_interest.classes_))

if st.button("🔍 Recommend Career"):
    try:
        # Encode
        skill_encoded = le_skills.transform([skill])[0]
        interest_encoded = le_interest.transform([interest])[0]

        # Predict
        prediction = model.predict([[gpa, skill_encoded, interest_encoded]])
        recommended_career = le_career.inverse_transform(prediction)[0]

        st.success(f"🎯 Recommended Career: **{recommended_career}**")

    except Exception as e:
        st.error(f"⚠️ Error occurred: {e}")
