import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- Chargement du label encoder ---
le = joblib.load("models/label_encoder.pkl")

# --- Sélecteur de modèle ---
st.title("Recommandation de Filière & Prédiction de Réussite")
model_choice = st.selectbox(
    "Choisissez le modèle à utiliser pour la prédiction :",
    ["Modèle de base ", "Modèle optimisé"]
)

model_path = "models/LogisticRegression.pkl" if "base" in model_choice else "models/GradientBoostingmeil.pkl"
model = joblib.load(model_path)

# --- Variables numériques ---
photoshop = st.slider("Maîtrise Photoshop (1-5)", 1, 5, 3)
creatif = st.slider("Résultat au test de créativité (0-10)", 0, 10, 5)
html_css = st.slider("Maîtrise HTML/CSS (1-5)", 1, 5, 3)
test_log = st.slider("Résultat au test de logique (0-10)", 0, 10, 5)
exp_avant = st.number_input("Années d'expérience avant reconversion", 0, 50, 1)
anglais = st.slider("Niveau d’anglais (0=élémentaire, 5=courant)", 0, 5, 3)
creatived = st.slider("Préférence logique vs. créatif (1=très créatif, 5=très logique)", 1, 5, 3)
equipe = st.slider("Préférence travail équipe vs solo (1=solo, 5=équipe)", 1, 5, 3)
note_dip = st.slider("Note moyenne du dernier diplôme (sur 20)", 0.0, 20.0, 12.0)

# --- Variable catégorielle Domaine ---
domaine_choices = [
    "Domaine littéraire", "Informatique / Développement logiciel",
    "Biologie / Sciences naturelles", "Marketing / Communication",
    "Sciences mathématiques / Physiques / Chimie", "BTP / Génie civile",
    "Réseaux et télécom", "Gestion de projet", "Autre"
]
domaine = st.selectbox("Domaine d'études précédent", domaine_choices)

# --- Variables multi-label ---
def multiselect_with_default(label, options):
    return st.multiselect(label, options)

langprog_options = ["Python", "JavaScript", "Java", "HTML / CSS", "Aucun (Néant)"]
bureautique_options = ["Microsoft Word", "Microsoft Excel", "Google Sheets", "Google Slides", "Powerpoint", "Jira", "Aucun (Néant)"]
design_options = ["Adobe Photoshop", "Canva", "Adobe Illustrator", "Aucun (Néant)", "Autre"]
transv_options = ["Gestion de projet", "Rédaction documentaire", "Gestion du stress", "Autonomie", "Proactivité", "Aucune (Néant)", "Autre"]
motivation_options = ["Passion et intérêt personnel", "Motivations professionnelles", "Motivations économiques", "Développement ou renforcement des compétences"]

langprog = multiselect_with_default("Langages de programmation connus", langprog_options)
bureautique = multiselect_with_default("Compétences bureautiques connues", bureautique_options)
design = multiselect_with_default("Outils de design maîtrisés", design_options)
transv = multiselect_with_default("Compétences transversales connues", transv_options)
motivation = multiselect_with_default("Motivation principale pour la reconversion", motivation_options)

# --- Encodage des multi-labels ---
def encode_multilabel(selection, possible_values, prefix):
    return {f"{prefix}_{val}": 1 if val in selection else 0 for val in possible_values}

# --- Construire l'entrée ---
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

# --- Prédiction ---
if st.button("Obtenir une recommandation"):
    try:
        prediction_num = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        confidence = round(np.max(proba) * 100, 2)
        prediction_label = le.inverse_transform([prediction_num])[0]

        st.success(f"Filière recommandée : **{prediction_label}**")


        st.info(f"Taux de réussite estimé : **{confidence}%**")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
