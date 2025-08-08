import os
import numpy as np
from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# --- App Initialization & Model Loading ---
app = Flask(__name__)

# --- NEW: Career Descriptions Dictionary ---
CAREER_DESCRIPTIONS = {
    "Android Developer": "Specializes in designing and building applications for mobile devices running the Android operating system. Key skills include Java, Kotlin, and the Android SDK.",
    "Backend Developer": "Focuses on the server-side of web applications. They work with databases, server logic, and APIs to ensure the application runs smoothly. Key skills include Python, Java, Node.js, and SQL.",
    "Business Analyst": "Acts as a bridge between business stakeholders and the IT team. They analyze business processes and data to identify areas for improvement. Key skills include SQL, Tableau, and strong communication.",
    "Cloud Engineer": "Designs, builds, and manages cloud-based infrastructure and applications on platforms like AWS, Azure, or GCP. Key skills include Docker, Kubernetes, and cloud service knowledge.",
    "Competitive Programmer": "Focuses on solving complex algorithmic problems under pressure, often for sport. This path builds deep problem-solving skills valuable in many software engineering roles. Key skills include C++, Java, and algorithms.",
    "Cybersecurity Analyst": "Protects an organization's computer systems and networks from cyber threats. They monitor for security breaches and implement security measures. Key skills include Nmap, Wireshark, and Python.",
    "Data Analyst": "Collects, cleans, and analyzes data to extract meaningful insights and help organizations make better decisions. Key skills include Python (Pandas), SQL, and data visualization tools like Excel or Tableau.",
    "Devops Engineer": "Works on automating and streamlining the software development and deployment process, bridging the gap between development and operations teams. Key skills include Docker, Jenkins, and cloud platforms.",
    "Frontend Developer": "Builds the visual and interactive parts of a website that users see and interact with directly in their browser. Key skills include HTML, CSS, JavaScript, and frameworks like React or Vue.js.",
    "Full Stack Developer": "A versatile developer who is comfortable working on both the frontend (client-side) and backend (server-side) of an application. Key skills include a mix of frontend and backend technologies.",
    "Game Developer": "Designs and develops video games for various platforms. This involves programming game mechanics, physics, and graphics. Key skills include C++, C#, and game engines like Unity or Unreal Engine.",
    "Machine Learning Engineer": "Builds and deploys machine learning models to solve business problems. They work at the intersection of software engineering and data science. Key skills include Python, TensorFlow/PyTorch, and SQL.",
    "Ui/Ux Designer": "Focuses on creating user-friendly and visually appealing interfaces. UI (User Interface) is about the look, while UX (User Experience) is about how it feels to use. Key skills include Figma, Adobe XD, and user research."
}

# --- Load AI Models & Encoders ---
def load_resources():
    model = load(os.path.join('model', 'model.pkl'))
    le_skills = load(os.path.join('model', 'skills_encoder.pkl'))
    le_interest = load(os.path.join('model', 'interest_encoder.pkl'))
    le_career = load(os.path.join('model', 'career_encoder.pkl'))
    return model, le_skills, le_interest, le_career

model, le_skills, le_interest, le_career = load_resources()
skills_list = sorted(le_skills.classes_)
interests_list = sorted(le_interest.classes_)

# --- Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    error = None
    selected_values = {}
    plot_url = None
    description = None  # NEW: Variable for career description

    if request.method == "POST":
        # ... (Your existing try/except block for getting form data and handling errors)
        try:
            selected_values['gpa'] = request.form.get("gpa")
            selected_values['interest'] = request.form.get("interest")
            selected_values['skill'] = request.form.get("skill")

            if not selected_values['gpa'] or not selected_values['interest'] or not selected_values['skill']:
                raise ValueError("All fields are required.")

            gpa = float(selected_values['gpa'])

            if gpa < 4.0:
                error = "A GPA below 4.0 is too low for a meaningful recommendation."
            else:
                skill_encoded = le_skills.transform([selected_values['skill']])[0]
                interest_encoded = le_interest.transform([selected_values['interest']])[0]
                features = np.array([[gpa, skill_encoded, interest_encoded]])
                
                pred_probabilities = model.predict_proba(features)[0]
                top_3_indices = np.argsort(pred_probabilities)[-3:][::-1]
                top_3_careers = le_career.inverse_transform(top_3_indices)
                top_3_probs = pred_probabilities[top_3_indices]
                
                predictions = list(zip(top_3_careers, top_3_probs))
                
                # NEW: Get the description for the top career
                top_career_name = predictions[0][0]
                description = CAREER_DESCRIPTIONS.get(top_career_name, "No description available for this career.")
                
                # --- Generate Feature Importance Plot ---
                feature_importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': ['GPA', 'Skill', 'Interest'],
                    'Importance': feature_importances
                }).sort_values(by='Importance', ascending=True)

                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(importance_df['Feature'], importance_df['Importance'], color='#0ea5e9')
                ax.set_title('Feature Importance')
                ax.set_xlabel('Influence on Prediction')
                
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                plt.close(fig)

        except ValueError as ve:
            error = f"Invalid input: {ve}"
        except Exception as e:
            error = f"An unexpected error occurred: {e}"

    return render_template(
        "index.html",
        predictions=predictions,
        error=error,
        skills=skills_list,
        interests=interests_list,
        selected=selected_values,
        plot_url=plot_url,
        description=description # NEW: Pass description to the template
    )

@app.route("/explorer")
def explorer():
    # ... (Your /explorer route remains the same)
    df = pd.read_csv('dataset/career_data.csv')
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))
    career_counts = df['Recommended_Career'].value_counts()
    sns.barplot(x=career_counts.values, y=career_counts.index, ax=ax, palette='viridis')
    ax.set_title('Number of Profiles per Career')
    ax.set_xlabel('Count')
    ax.set_ylabel('Career')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    table_html = df.to_html(classes='table table-dark table-striped table-hover', index=False, border=0)
    return render_template("explorer.html", table=table_html, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)