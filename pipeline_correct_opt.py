import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# --- Fonction multi-hot encoding pour colonnes multi-label ---
def multi_label_binarizer(df, column_name, possible_labels):
    df[column_name] = df[column_name].fillna("")
    for label in possible_labels:
        df[f"{column_name}_{label}"] = df[column_name].apply(lambda x: int(label in [v.strip() for v in x.split(',')]))
    df.drop(columns=[column_name], inplace=True)
    return df

# --- Chargement des données ---
df = pd.read_excel("bdd final.xlsx")

# --- Renommage des colonnes ---
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
    "Êtes vous plutôt créatif ? Préférence logique vs. créatif (1=Très créatif, 5=Très logique) ": "Creatived",
    "Préférez vous travailler en équipe ? Travail en équipe vs solo (1=Solo, 5=Équipe)": "Equipe",
    "Quel a été votre Résultat au test de logique (0-10) ** Si vous avez eu accès aux résultats": "TestLog",
    "Quel était votre niveau de maîtrise du logiciel photoshop avant la formation ? Auto-évaluation Photoshop (1 à 5)": "Photoshop",
    "Quel était votre niveau de maîtrise de HTML avant la formation ? Auto-évaluation HTML/CSS (1 à 5)": "HTML",
    "Niveau de langue étrangère - anglais - avant la formation (1 élémentaire - 5 courant)": "Anglais",
    "Quelle formation avez vous suivie (filière/spécialité)": "Formation",
    "Quel a été votre Résultat au test de créativité  (0-10) ** Si vous avez eu accès aux résultats":"Creatif"
}, inplace=True)

# --- Multi-hot encoding ---
multi_choice_cols = {
    "LangProg": ["Python", "JavaScript", "Java", "HTML / CSS", "Aucun (Néant)"],
    "Bureautique": ["Microsoft Word", "Microsoft Excel", "Google Sheets", "Google Slides", "Powerpoint", "Jira", "Aucun (Néant)"],
    "Design": ["Adobe Photoshop", "Canva", "Adobe Illustrator", "Aucun (Néant)", "Autre"],
    "Transv": ["Gestion de projet", "Rédaction documentaire", "Gestion du stress", "Autonomie", "Proactivité", "Aucune (Néant)", "Autre"],
    "Motivation": ["Passion et intérêt personnel", "Motivations professionnelles", "Motivations économiques", "Développement ou renforcement des compétences"]
}
for col, labels in multi_choice_cols.items():
    df = multi_label_binarizer(df, col, labels)

# --- Features & target ---
features = ["Photoshop", "Creatif", "Domaine", "HTML", "TestLog", "ExpAvant", "Creatived", "Equipe", "Anglais", "NoteDip"]
for col, labels in multi_choice_cols.items():
    features.extend([f"{col}_{label}" for label in labels])

target = "Formation"
le = LabelEncoder()
df["Formation_num"] = le.fit_transform(df[target])

X = df[features]
y = df["Formation_num"]

# Categorical + numerical features
cat_features = ["Domaine"]
num_features = [col for col in features if col != "Domaine"]
X[cat_features] = X[cat_features].astype(str)

preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="mean"), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- Modèles avec grilles d’hyperparamètres ---
models_with_params = {
    "RandomForest": (RandomForestClassifier(class_weight="balanced"), {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20]
    }),
    "GradientBoosting": (GradientBoostingClassifier(), {
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.1, 0.01]
    }),
    "AdaBoost": (AdaBoostClassifier(), {
        "classifier__n_estimators": [50, 100],
        "classifier__learning_rate": [1.0, 0.5]
    }),
    "Bagging": (BaggingClassifier(), {
        "classifier__n_estimators": [10, 50]
    }),
    "KNN": (KNeighborsClassifier(), {
        "classifier__n_neighbors": [3, 5, 7]
    }),
    "SVC": (SVC(probability=True), {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["linear", "rbf"]
    }),
    "MLP": (MLPClassifier(max_iter=500), {
        "classifier__hidden_layer_sizes": [(50,), (100,)],
        "classifier__alpha": [0.0001, 0.001]
    }),
    "NaiveBayes": (GaussianNB(), {}),  # no hyperparams
    "LogisticRegression": (
    LogisticRegression(max_iter=1000, solver="liblinear"),  # liblinear supporte l1 et l2
    {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1", "l2"],
        "classifier__class_weight": [None, "balanced"],
        "classifier__max_iter": [300, 500, 1000]
    }
),

    "Ridge": (RidgeClassifier(class_weight="balanced"), {
        "classifier__alpha": [0.1, 1, 10]
    }),
}
# --- Vérifier la distribution des classes ---
import numpy as np

class_counts = np.bincount(y)
print("Effectifs par classe (après encodage):")
for i, count in enumerate(class_counts):
    print(f"Classe {i} → {count} échantillons")

# --- AJOUT : Logistic Regression ajustée avec SMOTE + k-fold CV ---
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.utils.class_weight import compute_class_weight

# Validation croisée k-fold
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# SMOTE avec k_neighbors réduit
smote = SMOTE(random_state=42, k_neighbors=3)

# Class weights manuels (optionnel en plus de "balanced")
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print("⚖️ Class weights calculés:", class_weight_dict)

# Pipeline avec SMOTE
log_reg_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", smote),
    ("classifier", LogisticRegression(max_iter=1000, solver="liblinear"))
])

# Grille d’hyperparamètres
param_grid_logreg = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__penalty": ["l1", "l2"],
    "classifier__class_weight": [None, "balanced", class_weight_dict],
    "classifier__max_iter": [300, 500, 1000]
}

# GridSearchCV avec k-fold CV
gs_logreg = GridSearchCV(
    log_reg_pipeline,
    param_grid_logreg,
    cv=cv_strategy,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=2
)

# Entraînement
gs_logreg.fit(X_train, y_train)

# Prédiction
y_pred_logreg = gs_logreg.predict(X_test)

# Sauvegarde du modèle ajusté
joblib.dump(gs_logreg.best_estimator_, "logistic_ajuste.pkl")

# Résultats
print("\n🎯 Meilleurs paramètres (logistic_ajuste):")
print(gs_logreg.best_params_)

print("\n📊 Scores sur le test set (logistic_ajuste):")
print("Accuracy :", accuracy_score(y_test, y_pred_logreg))
print("Precision:", precision_score(y_test, y_pred_logreg, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_logreg, average="weighted", zero_division=0))
print("F1-score :", f1_score(y_test, y_pred_logreg, average="weighted", zero_division=0))
print("\n📁 Modèle logistic_ajuste enregistré dans 'logistic_ajuste.pkl'")
