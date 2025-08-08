import streamlit as st

st.set_page_config(
    page_title="AI Career Recommender",
    page_icon="🤖",
    layout="wide"
)

st.title("Welcome to the AI Career Recommender! 🎓")
st.markdown("---")

st.header("Navigate Your Future Career Path")
st.write(
    "This dashboard is designed to help you explore potential career paths using the power of machine learning. "
    "Use the sidebar to navigate between pages."
)

st.info("👈 **Select a page from the sidebar to get started!**", icon="ℹ️")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚀 Career Recommender")
    st.write("Get a personalized career recommendation based on your GPA, skills, and interests. This page now includes prediction probabilities and explains why a recommendation was made.")
    
with col2:
    st.subheader("📊 Data Explorer")
    st.write("Interactively explore the dataset that was used to train our AI model. See distributions of careers, skills, and more.")

st.markdown("---")
st.image(
    "https://images.unsplash.com/photo-1552664730-d307ca884978?q=80&w=2070",
    caption="Let's find the right path for you."
)