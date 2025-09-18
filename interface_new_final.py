import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- Load label encoder ---
le = joblib.load("models/label_encoder.pkl")

# --- Model selector ---
st.title("Field Recommendation & Success Prediction")
model_choice = st.selectbox(
    "Choose the Logistic Regression model to use:",
    [
        "Logistic Regression (simple)",
        "Logistic Regression + SMOTE",
        "Logistic Regression + class_weight"
    ]
)

# --- Load model based on choice ---
model_paths = {
    "Logistic Regression (simple)": "models/LogisticRegression.pkl",
    "Logistic Regression + SMOTE": "models/LogisticRegression_smote.pkl",
    "Logistic Regression + class_weight": "models/logistic_class_weighted.pkl"
}

model = joblib.load(model_paths[model_choice])

# --- Numerical variables ---
photoshop = st.slider("Photoshop skills (1-5)", 1, 5, 3)
creatif = st.slider("Creativity test score (0-10)", 0, 10, 5)
html_css = st.slider("HTML/CSS skills (1-5)", 1, 5, 3)
test_log = st.slider("Logic test score (0-10)", 0, 10, 5)
exp_avant = st.number_input("Years of experience before career change", 0, 50, 1)
anglais = st.slider("English level (0=elementary, 5=fluent)", 0, 5, 3)
creatived = st.slider("Logical vs. creative preference (1=very creative, 5=very logical)", 1, 5, 3)
equipe = st.slider("Teamwork vs. solo preference (1=solo, 5=team)", 1, 5, 3)
note_dip = st.slider("Average grade of last diploma (out of 20)", 0.0, 20.0, 12.0)

# --- Previous field ---
domaine_choices = [
    "Literature / Humanities", "Computer Science / Software Development",
    "Biology / Natural Sciences", "Marketing / Communication",
    "Mathematics / Physics / Chemistry", "Civil Engineering / Construction",
    "Networks & Telecommunications", "Project Management", "Other"
]
domaine = st.selectbox("Previous field of study", domaine_choices)

# --- Multi-label variables ---
def multiselect_with_default(label, options):
    return st.multiselect(label, options)

langprog_options = ["Python", "JavaScript", "Java", "HTML / CSS", "None"]
bureautique_options = ["Microsoft Word", "Microsoft Excel", "Google Sheets", "Google Slides", "PowerPoint", "Jira", "None"]
design_options = ["Adobe Photoshop", "Canva", "Adobe Illustrator", "None", "Other"]
transv_options = ["Project management", "Documentation writing", "Stress management", "Autonomy", "Proactivity", "None", "Other"]
motivation_options = ["Personal passion and interest", "Professional motivations", "Economic motivations", "Skill development / improvement"]

langprog = multiselect_with_default("Programming languages known", langprog_options)
bureautique = multiselect_with_default("Office tools known", bureautique_options)
design = multiselect_with_default("Design tools mastered", design_options)
transv = multiselect_with_default("Transversal skills known", transv_options)
motivation = multiselect_with_default("Main motivation for career change", motivation_options)

# --- Multi-label encoding ---
def encode_multilabel(selection, possible_values, prefix):
    return {f"{prefix}_{val}": 1 if val in selection else 0 for val in possible_values}

# --- Build input ---
input_dict = {
    "Photoshop": photoshop,
    "Creatif": creatif,
    "Domaine": domaine,
    "HTML": html_css,
    "TestLog": test_log,
    "ExpAvant": exp_avant,
    "Creatived": creatived,
    "Equipe": equipe,
    "Anglais": anglais,
    "NoteDip": note_dip,
}
input_dict.update(encode_multilabel(langprog, langprog_options, "LangProg"))
input_dict.update(encode_multilabel(bureautique, bureautique_options, "Bureautique"))
input_dict.update(encode_multilabel(design, design_options, "Design"))
input_dict.update(encode_multilabel(transv, transv_options, "Transv"))
input_dict.update(encode_multilabel(motivation, motivation_options, "Motivation"))

X_input = pd.DataFrame([input_dict])

# --- Prediction ---
if st.button("Get Recommendation"):
    try:
        prediction_num = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        confidence = round(np.max(proba) * 100, 2)
        prediction_label = le.inverse_transform([prediction_num])[0]

        st.success(f"Recommended field: **{prediction_label}**")
        st.info(f"Estimated success rate: **{confidence}%**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
