# 🧠 Crime Risk Predictor

A psychological trait-based **Crime Risk Prediction System** built with **XGBoost** and **Streamlit**, designed to estimate the likelihood of an individual engaging in criminal behavior. This project emphasizes ethical AI practices and is intended for academic and educational use only.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [How It Works](#-how-it-works)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## 📖 Overview

This project is a machine learning-powered web application that predicts the **risk of criminal behavior** based on psychological traits and demographic data. It is built using **XGBoost** for classification and **SHAP** for interpretability, with an interactive frontend developed using **Streamlit**.

### ✅ Key Features

- 🔢 Accepts **35+ psychological and demographic traits** as input via sliders and selectors
- 🤖 Utilizes **XGBoost classifier** for robust and efficient prediction
- 📊 Provides **probability-based crime risk score** (0% to 100%)
- 📌 Highlights **top contributing traits** using SHAP values for model transparency
- 📈 Includes an **Insights** page with:
  - Trait distribution plots (bar graphs)
  - Correlation heatmap
  - Global SHAP summary plot
- 🧠 Visual explanation of each prediction using **SHAP force plots** and **bar plots**
- 🔒 Includes **ethical disclaimer** for responsible usage

---

## ✅ Features

- 🔢 **Manual Input Form**: Enter traits & demographic data
- 📊 **SHAP Visuals**: Understand which features drive risk
- 🌐 **Insights Dashboard**: Visualize feature distributions and correlations
- 🧠 **Explainable ML**: Shows impact of individual traits
- 🎓 **Educational Tool**: For psychology, criminology, or ML research

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit** – Web app UI
- **XGBoost** – Classification model
- **SHAP** – Explainable AI
- **Pandas**, **Matplotlib**, **Seaborn** – Data analysis & visualization

---

## 🚀 Demo

You can run the app locally or deploy it on [Streamlit Cloud](https://streamlit.io/cloud).

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/mahalakshmi118/crime-risk-predictor.git
cd crime-risk-predictor
```

### 2. (Optional) Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
crime-risk-predictor/
│
├── crime.ipynb               # Development notebook
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── artifacts/                # Saved models and datasets
│   ├── xgb_model.pkl
│   ├── X.pkl
│   ├── X_test.pkl
│   └── y_test.pkl
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
It is **not intended** for real-world judicial, behavioral, or psychological assessment.  
Trait-based profiling can be **subjective and sensitive** — use responsibly and ethically.

---

## 📄 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and share with attribution.
