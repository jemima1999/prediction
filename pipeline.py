import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer

# 📥 Chargement
df = pd.read_excel("bdd final.xlsx")

# 🧼 Renommage des colonnes
df.rename(columns={
    "Âge": "Age", "Genre": "Genre", "Pays de provenance": "Pays",
    "Domaine d’études avant la reconversion": "Domaine",
    "Niveau d’études le plus élevé avant la reconversion": "Niveau",
    "Note (moyenne) de sortie du diplôme le plus élevé avant la formation": "NoteDip",
    "Année d’obtention du diplôme le plus élevé avant la formation": "AnneeDip",
    "Moyenne au BAC/DT si vous l'avez": "MoyBAC",
    "Moyenne au BEPC/CAP si vous l'avez": "MoyBEPC",
    "Secteur d’activité avant la formation si professionnelle": "SecteurAvant",
    "Nombre d'années d’expérience avant la reconversion / formation": "ExpAvant",
    "Langages de programmation connus avant la formation/reconversion": "LangProg",
    "Quelle a été votre motivation principale ou l'élément déclencheur pour la reconversion (expliqué en détail)": "Motivation",
    "Outils de design maîtrisés avant la formation/reconversion": "Design",
    "Compétences bureautiques avant la formation/reconversion": "Bureautique",
    "Compétences transversales avant la formation/reconversion": "Transv",
    "Aviez vous un ordinateur personnel durant la formation ?": "Ordi",
    "Aviez vous accès à une Connexion Internet stable chez vous à la maison ?": "Internet",
    "Êtes vous plutôt créatif ? Préférence logique vs. créatif (1=Très créatif, 5=Très logique) ": "Creatif",
    "Préférez vous travailler en équipe ? Travail en équipe vs solo (1=Solo, 5=Équipe)": "Equipe",
    "Quel a été votre Résultat au test de logique (0-10) ** Si vous avez eu accès aux résultats": "TestLog",
    "Quel était votre niveau de maîtrise du logiciel photoshop avant la formation ? Auto-évaluation Photoshop (1 à 5)": "Photoshop",
    "Quel était votre niveau de maîtrise de HTML avant la formation ? Auto-évaluation HTML/CSS (1 à 5)": "HTML",
    "Niveau de langue étrangère - anglais - avant la formation (1 élémentaire - 5 courant)": "Anglais",
    "Quelle formation avez vous suivie (filière/spécialité)": "Formation"
}, inplace=True)

# ✅ Variables utilisées
features = ["Photoshop", "Creatif", "Domaine", "HTML", "TestLog", "ExpAvant", "Transv",
            "AnneeDip", "Design", "Bureautique", "Motivation", "Anglais", "LangProg",
            "Equipe", "NoteDip"]

target = "Formation"

# 🎯 Encodage de la cible
le = LabelEncoder()
df["Formation_num"] = le.fit_transform(df[target])

print("Colonnes disponibles :", df.columns.tolist())

# ⚠️ Conversion des colonnes catégoriques en str pour éviter mélange types dans OneHotEncoder
X = df[features]

# Séparation num / cat
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Conversion forcée en str des colonnes catégoriques
for col in cat_features:
    X[col] = X[col].astype(str)

y = df["Formation_num"]

# 🔧 Prétraitement
preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="mean"), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# 📦 Modèles
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "Bagging": BaggingClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True, random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=500),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "Ridge": RidgeClassifier(class_weight="balanced"),
}

# ➕ Stacking model
stacking_model = StackingClassifier(
    estimators=[
        ("rf", models["RandomForest"]),
        ("gb", models["GradientBoosting"])
    ],
    final_estimator=LogisticRegression(class_weight="balanced")
)
models["Stacking"] = stacking_model

# 🔁 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 📁 Création du dossier de sauvegarde
os.makedirs("models", exist_ok=True)

# 🔁 Entraînement et sauvegarde
for name, clf in models.items():
    try:
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipeline.fit(X_train, y_train)

        # Sauvegarde du pipeline complet
        joblib.dump(pipeline, f"models/{name}.pkl")
        print(f"✅ Modèle {name} sauvegardé avec succès.")

    except Exception as e:
        print(f"❌ Erreur avec le modèle {name} : {e}")

