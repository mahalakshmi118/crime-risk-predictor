
# 🔧 Must be first!
import streamlit as st
st.set_page_config(page_title="Crime Risk Predictor", layout="centered")
# 📦 Imports
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
# 🔁 Cached Data Load
@st.cache_data
def load_data():
    X = joblib.load("artifacts/X.pkl")
    X_test = joblib.load("artifacts/X_test.pkl")
    y_test = joblib.load("artifacts/y_test.pkl")
    model = joblib.load("artifacts/xgb_model.pkl")
    return X, X_test, y_test, model
X, X_test_df, y_test, model = load_data()
# 📋 Feature List
features_order = [
    'antisocial_traits', 'poor_impulse_control', 'low_self_esteem', 'manipulative_tendencies',
    'aggression', 'social_withdrawal', 'childhood_trauma', 'emotional_instability',
    'peer_pressure_susceptibility', 'risk_taking', 'narcissism', 'callous_unemotional_traits',
    'borderline_personality_features', 'paranoia', 'cognitive_distortions', 'lack_of_empathy',
    'compulsive_behavior', 'thrill_seeking', 'unrealistic_goals', 'blame_externalization',
    'rationalization', 'lack_of_guilt', 'envy', 'need_for_power', 'need_for_approval',
    'jealousy', 'attachment_disorder', 'empathy_deficit', 'emotional_blunting', 'passive_aggression',
    'impulsivity', 'emotional_dysregulation', 'detachment', 'aggression_violence',
    'age', 'education_level', 'employment_status'
]
# 📍 Navigation
page = st.sidebar.radio("Navigation", ["🏠 Predictor", "📊 Insights"])
# 📊 Insights Page
if page == "📊 Insights":
    st.title("📊 Trait Distributions & Correlations")
    st.write("Understand how psychological traits vary across individuals.")
    # 🔹 Bar charts for top 5 traits
    st.subheader("🔢 Trait Distributions")
    st.markdown("Each bar plot below shows how values for a trait are distributed among individuals (grouped into bins).")
    
    for col in X.columns[:5]:
        # Group values into bins for clearer visualization
        counts = pd.cut(X[col], bins=10).value_counts().sort_index()
        
        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(f'Distribution of {col.replace("_", " ").title()}', fontsize=12)
        ax.set_xlabel('Value Range')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    # 🔸 Correlation heatmap
    st.subheader("📈 Correlation Heatmap")
    st.markdown("Visualizing the correlation between different traits.")
    fig_corr, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    st.pyplot(fig_corr)
    # 🔸 Global SHAP using TreeExplainer
    st.subheader("🌐 Global Feature Importance (SHAP)")
    st.markdown("Top features influencing model predictions, based on SHAP values.")
    sample_X = X_test_df.sample(n=100, random_state=42)  # sample for speed
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer.shap_values(sample_X)
    fig_summary, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values_all, sample_X, show=False)
    st.pyplot(fig_summary)
# 🏠 Predictor Page
if page == "🏠 Predictor":
    st.title("🧠 Psychological Trait-Based Crime Risk Predictor")
    st.markdown("This model predicts an individual's likelihood of criminal behavior based on **psychological traits** and **demographics**.")
    # 🧾 Input Form
    data = {}
    st.header("📋 Input Traits & Demographics")
    with st.form("trait_form"):
        for feat in features_order:
            if feat == 'education_level':
                data[feat] = st.slider("Education Level (1=No schooling to 5=Higher education)", 1, 5, 3)
            elif feat == 'employment_status':
                data[feat] = st.radio("Employment Status", [0, 1], format_func=lambda x: "Unemployed" if x == 0 else "Employed")
            elif feat == 'age':
                data[feat] = st.slider("Age", 18, 60, 30)
            else:
                data[feat] = st.slider(feat.replace("_", " ").title(), 0.0, 1.0, 0.5, 0.01)
        submitted = st.form_submit_button("🔮 Predict Crime Risk")
    if submitted:
        # 🎯 Predict
        X_input = pd.DataFrame([data])[features_order]
        prob = model.predict_proba(X_input)[0][1]
        percent = round(prob * 100, 2)
        # 🎯 Result
        st.subheader("🔎 Prediction Result")
        color = "🟢" if percent < 40 else "🟡" if percent < 70 else "🔴"
        st.markdown(f"### {color} Risk Score: **{percent}%**")
        # 🧠 SHAP for single prediction
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)
        impact = shap_values[0]
        # 🔝 Top features
        top_idx = np.argsort(np.abs(impact))[::-1][:5]
        st.subheader("📌 Top Factors Contributing to Risk")
        for idx in top_idx:
            f = features_order[idx]
            val = data[f]
            direction = "⬆️ Increase" if impact[idx] > 0 else "⬇️ Decrease"
            st.markdown(f"**{f.replace('_', ' ').title()}**: {val} → {direction} risk (Impact: {impact[idx]:.4f})")
        # 📊 SHAP bar plot
        st.subheader("📊 Feature Importance Visualized")
        explanation = shap.Explanation(values=impact, data=X_input.values[0], feature_names=features_order)
        fig_bar, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(explanation, show=False)
        st.pyplot(fig_bar)
        # 🧠 Explanation
        st.subheader("📘 Explanation")
        st.markdown(f"""
        The predicted crime risk of **{percent}%** is derived from a combination of psychological and demographic traits.
        Traits like **antisocial behavior**, **impulsivity**, or **lack of empathy** may contribute more.
        Positive influences like **employment**, **education**, or **age** can lower the risk.
        """)
        st.markdown("🔒 This model is for **educational purposes only** and should not be used in real-world assessments.")
