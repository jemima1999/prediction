import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Load label encoder ---
le = joblib.load("models/label_encoder.pkl")

# --- Model selector ---
st.title("Program Recommendation & Success Prediction")
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

# --- Numeric variables ---
photoshop = st.slider("Photoshop proficiency (1-5)", 1, 5, 3)
creatif = st.slider("Creativity test score (0-10)", 0, 10, 5)
html_css = st.slider("HTML/CSS proficiency (1-5)", 1, 5, 3)
test_log = st.slider("Logic test score (0-10)", 0, 10, 5)
exp_avant = st.number_input("Years of experience before career change", 0, 50, 1)
anglais = st.slider("English level (0=basic, 5=fluent)", 0, 5, 3)
creatived = st.slider("Logic vs. creativity preference (1=very creative, 5=very logical)", 1, 5, 3)
equipe = st.slider("Team vs. solo work preference (1=solo, 5=team)", 1, 5, 3)
note_dip = st.slider("Average grade of last diploma (out of 20)", 0.0, 20.0, 12.0)

# --- Categorical domain ---
domaine_choices = [
    "Domaine littéraire", "Informatique / Développement logiciel",
    "Biologie / Sciences naturelles", "Marketing / Communication",
    "Sciences mathématiques / Physiques / Chimie", "BTP / Génie civile",
    "Réseaux et télécom", "Gestion de projet", "Autre"
]
domaine_labels = [
    "Literature field", "Computer Science / Software Development",
    "Biology / Natural Sciences", "Marketing / Communication",
    "Mathematics / Physics / Chemistry", "Construction / Civil Engineering",
    "Networking and Telecom", "Project Management", "Other"
]
domaine = st.selectbox("Previous field of study", options=domaine_choices, format_func=lambda x: domaine_labels[domaine_choices.index(x)])

# --- Multi-label variables ---
def multiselect_with_default(label, options, labels):
    return st.multiselect(label, options, format_func=lambda x: labels[options.index(x)])

langprog_options = ["Python", "JavaScript", "Java", "HTML / CSS", "Aucun (Néant)"]
langprog_labels = ["Python", "JavaScript", "Java", "HTML / CSS", "None"]

bureautique_options = ["Microsoft Word", "Microsoft Excel", "Google Sheets", "Google Slides", "Powerpoint", "Jira", "Aucun (Néant)"]
bureautique_labels = ["Microsoft Word", "Microsoft Excel", "Google Sheets", "Google Slides", "Powerpoint", "Jira", "None"]

design_options = ["Adobe Photoshop", "Canva", "Adobe Illustrator", "Aucun (Néant)", "Autre"]
design_labels = ["Adobe Photoshop", "Canva", "Adobe Illustrator", "None", "Other"]

transv_options = ["Gestion de projet", "Rédaction documentaire", "Gestion du stress", "Autonomie", "Proactivité", "Aucune (Néant)", "Autre"]
transv_labels = ["Project Management", "Documentation", "Stress Management", "Autonomy", "Proactivity", "None", "Other"]

motivation_options = ["Passion et intérêt personnel", "Motivations professionnelles", "Motivations économiques", "Développement ou renforcement des compétences"]
motivation_labels = ["Passion and personal interest", "Professional motivations", "Economic motivations", "Skill development or enhancement"]

langprog = multiselect_with_default("Known programming languages", langprog_options, langprog_labels)
bureautique = multiselect_with_default("Known office skills", bureautique_options, bureautique_labels)
design = multiselect_with_default("Design tools mastered", design_options, design_labels)
transv = multiselect_with_default("Known transversal skills", transv_options, transv_labels)
motivation = multiselect_with_default("Main motivation for career change", motivation_options, motivation_labels)

# --- Encode multi-labels ---
def encode_multilabel(selection, possible_values, prefix):
    return {f"{prefix}_{val}": 1 if val in selection else 0 for val in possible_values}

# --- Build input dict ---
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

# --- Label translation dictionary ---
label_translation = {
    "Gestionnaire de projet digital": "Digital Project Manager",
    "Designer graphique et numérique": "Graphic & Digital Designer",
    "Spécialiste Audiovisuel": "Audiovisual Specialist",
    "Développeur d'applications": "Application Developer",
    "Autres": "Other"
}

# --- Prediction ---
if st.button("Get Recommendation"):
    try:
        prediction_num = model.predict(X_input)[0]

        proba = model.predict_proba(X_input)[0]
        confidence = round(np.max(proba) * 100, 2)
        prediction_label_fr = le.inverse_transform([prediction_num])[0].strip()
     
        
        
        

        # Translate prediction to English
        prediction_label_en = label_translation.get(prediction_label_fr, prediction_label_fr)

        st.success(f"Recommended program: **{prediction_label_en}**")
        st.info(f"Estimated success rate: **{confidence}%**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
