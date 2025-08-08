import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Explorer", layout="wide")

st.title("📊 Data Explorer")
st.write("Explore the dataset used to train the recommendation model.")

@st.cache_data
def load_data():
    """Loads the career dataset."""
    df = pd.read_csv('dataset/career_data.csv')
    return df

df = load_data()

st.markdown("### 📈 Visualizations")
chart_type = st.selectbox(
    "Choose a chart to display:",
    ["Distribution of Recommended Careers", "GPA Distribution by Interest", "Skills by Interest"]
)

if chart_type == "Distribution of Recommended Careers":
    career_counts = df['Recommended_Career'].value_counts().reset_index()
    career_counts.columns = ['Career', 'Count']
    fig = px.bar(
        career_counts, 
        x='Count', 
        y='Career', 
        orientation='h', 
        title='Number of Profiles per Career',
        text='Count'
    )
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "GPA Distribution by Interest":
    fig = px.box(
        df, 
        x='Interest', 
        y='GPA', 
        title='GPA Distribution across different Interests'
    )
    st.plotly_chart(fig, use_container_width=True)

elif chart_type == "Skills by Interest":
    skills_by_interest = df.groupby('Interest')['Skills'].nunique().reset_index()
    fig = px.pie(
        skills_by_interest, 
        names='Interest', 
        values='Skills', 
        title='Number of Unique Skills per Interest Area'
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Raw Dataset")
st.dataframe(df)