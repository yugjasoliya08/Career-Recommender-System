import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
import os

# Load dataset
df = pd.read_csv("dataset/career_data.csv")

# Initialize label encoders
le_skills = LabelEncoder()
le_interest = LabelEncoder()
le_career = LabelEncoder()

# Encode categorical columns
df["Skills"] = le_skills.fit_transform(df["Skills"])
df["Interest"] = le_interest.fit_transform(df["Interest"])
df["Recommended_Career"] = le_career.fit_transform(df["Recommended_Career"])

# Split features and target
X = df[["GPA", "Skills", "Interest"]]
y = df["Recommended_Career"]

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Create model directory if not exists
os.makedirs("model", exist_ok=True)

# Save model and encoders
dump(model, "model/model.pkl")
dump(le_skills, "model/skills_encoder.pkl")
dump(le_interest, "model/interest_encoder.pkl")
dump(le_career, "model/career_encoder.pkl")

# Output info
print("\n✅ Model training complete. All files saved in 'model/' folder.\n")
print("🛠 Available Skills: ", list(le_skills.classes_))
print("🎯 Available Interests: ", list(le_interest.classes_))
print("🎓 Target Careers: ", list(le_career.classes_))
