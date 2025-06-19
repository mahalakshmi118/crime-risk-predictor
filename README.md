# 🧠 Crime Risk Predictor

A psychological trait-based **Crime Risk Prediction System** built with **XGBoost** and **Streamlit**, designed to estimate the likelihood of an individual engaging in criminal behavior. This project emphasizes ethical AI practices and is intended for academic and educational use only.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [How It Works](#how-it-works)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## 📖 Overview

This project is a machine learning-powered web app that predicts crime risk using psychological traits and demographic features such as:

- Antisocial tendencies
- Aggression
- Impulsivity
- Childhood trauma
- Education and employment status

The model uses **XGBoost** for classification and **SHAP** values for interpreting the model’s predictions.

---

## ✅ Features

- 🔢 **Manual Input Form**: Enter traits & demographic data.
- 📊 **SHAP Visuals**: Understand which features drive risk.
- 🌐 **Insights Dashboard**: Visualize feature distributions and correlations.
- 🧠 **Explainable ML**: Shows impact of individual traits.
- 🎓 **Educational Tool**: Intended for psychological, criminology, or ML research purposes.

---

## 🛠️ Technologies Used

- Python 🐍
- Streamlit 📺 (for web app UI)
- XGBoost 🌲 (classification model)
- SHAP 🧠 (explainable AI)
- Pandas, Matplotlib, Seaborn 📈 (data analysis & visualization)

---

## 🚀 Demo

You can run the app locally or deploy on [Streamlit Cloud](https://streamlit.io/cloud).

---

## 🔧 Installation

1. **Clone the repo**:

```bash
git clone https://github.com/mahalakshmi118/crime-risk-predictor.git
cd crime-risk-predictor

2. **Create and activate a virtual environment (optional but recommended)**:

```bash
Copy
Edit
python -m venv venv
venv\Scripts\activate   # On Windows

3. **Install dependencies**:

```bash
Copy
Edit
pip install -r requirements.txt

4. **Run the Streamlit app**:

```bash
Copy
Edit
streamlit run app.py

## 📁 Project Structure

crime-risk-predictor/
│
├── artifacts/               # Saved models and datasets
│   ├── xgb_model.pkl
│   ├── X.pkl
│   ├── X_test.pkl
│   └── y_test.pkl
│
├── app.py                   # Main Streamlit app
├── requirements.txt         # Dependencies
└── README.md                # Project documentation

⚠️ Disclaimer
This project is for educational purposes only.

It is not intended or suitable for real-world judicial, medical, or behavioral assessment use. Psychological trait data can be subjective and highly sensitive.

📄 License
MIT License
Feel free to use, modify, and share with attribution.
