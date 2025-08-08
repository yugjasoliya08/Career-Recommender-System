import streamlit as st
from joblib import load
import numpy as np
import os
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Career Recommender", layout="wide")

# --- Load Models ---
@st.cache_resource
def load_models():
    """Loads the trained model and encoders from disk."""
    model = load(os.path.join('model', 'model.pkl'))
    le_skills = load(os.path.join('model', 'skills_encoder.pkl'))
    le_interest = load(os.path.join('model', 'interest_encoder.pkl'))
    le_career = load(os.path.join('model', 'career_encoder.pkl'))
    return model, le_skills, le_interest, le_career

model, le_skills, le_interest, le_career = load_models()
skills_options = sorted(le_skills.classes_)
interests_options = sorted(le_interest.classes_)

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("👤 Your Profile")
    st.markdown("Tell us about yourself.")
    gpa = st.slider("Select your GPA:", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    skill = st.selectbox("Select your Top Skill:", skills_options)
    interest = st.selectbox("Select your Primary Interest:", interests_options)
    st.markdown("---")
    predict_button = st.button("Recommend Career", type="primary", use_container_width=True)

# --- Main Page Content ---
st.title("🚀 Career Recommendation Engine")

if predict_button:
    # Check for a realistic GPA before predicting
    if gpa < 4.0:
        # --- THIS LINE IS FIXED ---
        st.error("A GPA below 4.0 is too low for a meaningful recommendation. Please select a higher GPA.", icon="📉")
    else:
        # Existing prediction logic
        try:
            skill_encoded = le_skills.transform([skill])[0]
            interest_encoded = le_interest.transform([interest])[0]
            features = np.array([[gpa, skill_encoded, interest_encoded]])
            
            pred_probabilities = model.predict_proba(features)[0]
            
            st.header("✨ Your Personalized Recommendation")

            top_3_indices = np.argsort(pred_probabilities)[-3:][::-1]
            top_3_careers = le_career.inverse_transform(top_3_indices)
            top_3_probs = pred_probabilities[top_3_indices]

            st.success(f"### 🎯 Top Recommendation: **{top_3_careers[0]}**")
            st.write(f"Confidence Score: **{top_3_probs[0]:.2%}**")

            st.subheader("Other Potential Paths:")
            for i in range(1, len(top_3_careers)):
                st.write(f"{i+1}. **{top_3_careers[i]}** (Confidence: {top_3_probs[i]:.2%})")

            st.markdown("---")
            st.subheader("💡 Why This Recommendation?")
            st.write("This chart shows how much each of your inputs influenced the model's decision.")

            feature_importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': ['GPA', 'Skill', 'Interest'],
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)
            
            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                text=importance_df['Importance'].apply(lambda x: f'{x:.2f}'),
                title="Feature Importance"
            )
            fig.update_layout(yaxis_title="Your Inputs", xaxis_title="Influence on Prediction")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction. Please try again. \n\nDetails: {e}")
else:
    st.info("Please fill out your profile in the sidebar and click the **'Recommend Career'** button to see your result.")