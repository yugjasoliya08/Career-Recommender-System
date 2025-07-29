# 🎓 Career Recommender System

A simple and effective **Career Recommender System** built using **Python**, **Flask**, and **Machine Learning**.

---

## 🚀 Project Overview

This project predicts the most suitable career path for a user based on:
- 📊 **GPA**
- 💡 **Skills**
- ❤️ **Interest**

It uses a Machine Learning model (Decision Tree) trained on a custom `career_data.csv` dataset.

---

## 🛠️ Tech Stack

- **Python 3**
- **Pandas**
- **Scikit-learn**
- **Flask** (Web Framework)
- **HTML/CSS** (Basic frontend)

---

## 📁 Project Structure

Career Recommender System/
│
├── dataset/ # Contains dataset files (e.g., career_data.csv)
├── model/ # Stores the trained model (e.g., model.pkl)
├── templates/ # HTML templates (index.html, result.html)
│
├── app.py # Flask backend server
├── career_gui.py # Optional GUI version (Tkinter or other)
├── predict_career.py # Handles prediction logic
├── train_model.py # Script to train and save ML model
├── README.md # Project documentation
├── .gitignore # Git ignore file

## 🚀 Features

- 🔍 Input GPA, skills, and interest
- 🧠 Uses Decision Tree ML model
- 🌐 Flask-based web UI
- 📊 Simple and extensible dataset
- 🖥️ Also includes a GUI version (`career_gui.py`)

---

## 🛠️ Technologies Used

- Python 3
- Flask
- Pandas
- scikit-learn
- HTML / Jinja2 (Templates)

---

## ▶️ How to Run

### 1. Clone the Repository

git clone https://github.com/your-username/Career-Recommender-System.git
cd Career-Recommender-System

### 2. Install Required Packages
Create requirements.txt if not present:
pip install flask pandas scikit-learn
Or:
pip install -r requirements.txt

### 3. Train the Model
python train_model.py

### 4. Run the Flask App
python app.py
Visit http://127.0.0.1:5000 in your browser.

##🧪 Dataset Format
CSV file inside /dataset/ folder (e.g., career_data.csv):

GPA,Skills,Interest,Recommended_Career
8.1,Python;ML,AI,Machine Learning Engineer
7.5,Java;SQL,Backend,Backend Developer

##🧠 How It Works

User submits GPA, skills, and interests via the form.
Data is passed to the backend ML model.
Trained Decision Tree predicts the best-fit career.
Result is displayed to the user.

##✨ Screenshots 

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/dfebcab4-7638-4760-ac84-64440f978f02" />


##👤 Author
Yug Jasoliya
📧 [yugjasoliya49@gmail.com]
🔗 linkdin:https://www.linkedin.com/in/yug-jasoliya-69691126b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app


