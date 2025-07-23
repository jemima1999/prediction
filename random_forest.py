import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Charger les données
df = pd.read_excel("bdd final.xlsx")

# Renommer les colonnes
new_names = {
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
    "Êtes vous plutôt créatif ? Préférence logique vs. créatif (1=Très créatif, 5=Très logique)": "Creatif",
    "Préférez vous travailler en équipe ? Travail en équipe vs solo (1=Solo, 5=Équipe)": "Equipe",
    "Quel a été votre Résultat au test de logique (0-10) ** Si vous avez eu accès aux résultats": "TestLog",
    "Quel était votre niveau de maîtrise du logiciel photoshop avant la formation ? Auto-évaluation Photoshop (1 à 5)": "Photoshop",
    "Quel était votre niveau de maîtrise de HTML avant la formation ? Auto-évaluation HTML/CSS (1 à 5)": "HTML",
    "Niveau de langue étrangère - anglais - avant la formation (1 élémentaire - 5 courant)":"Anglais",
    "Quelle formation avez vous suivie (filière/spécialité)": "Formation"
}
df.rename(columns=new_names, inplace=True)

# Liste finale complète des variables (toutes les importantes)
final_features = ["Photoshop","Creatif","Domaine","HTML","TestLog","ExpAvant","Transv","AnneeDip","Design","Bureautique","Motivation","Anglais","Creatif","LangProg","Equipe","NoteDip",]


target = "Formation"
print(df["Formation"].value_counts())

# Encoder la cible
le = LabelEncoder()
df["Formation_num"] = le.fit_transform(df[target])

# Conserver seulement les colonnes qui existent dans le dataframe (protection)
final_features = [feat for feat in final_features if feat in df.columns]

X = df[final_features]
y = df["Formation_num"]

# Encoder toutes les variables catégorielles automatiquement
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modèle Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42,class_weight="balanced")
model.fit(X_train, y_train)

# Scores
print("Score train :", model.score(X_train, y_train))
print("Score test :", model.score(X_test, y_test))

# Prédictions
y_pred = model.predict(X_test)

# Classes réellement présentes dans y_test
import numpy as np
labels_present = np.unique(y_test)
target_names_present = le.inverse_transform(labels_present)

# Classification report
print("\nClassification report :")
print(classification_report(y_test, y_pred, labels=labels_present, target_names=target_names_present))

# Matrice de confusion
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))
