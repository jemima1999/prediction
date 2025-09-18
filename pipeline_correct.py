import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

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
    "Âge": "Age", 
    "Genre": "Genre", 
    "Pays de provenance": "Pays",
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

# --- Définition des colonnes multi-label et leurs valeurs possibles ---
multi_choice_cols = {
    "LangProg": ["Python", "JavaScript", "Java", "HTML / CSS", "Aucun (Néant)"],
    "Bureautique": ["Microsoft Word", "Microsoft Excel", "Google Sheets", "Google Slides", "Powerpoint", "Jira", "Aucun (Néant)"],
    "Design": ["Adobe Photoshop", "Canva", "Adobe Illustrator", "Aucun (Néant)", "Autre"],
    "Transv": ["Gestion de projet", "Rédaction documentaire", "Gestion du stress", "Autonomie", "Proactivité", "Aucune (Néant)", "Autre"],
    "Motivation": ["Passion et intérêt personnel", "Motivations professionnelles", "Motivations économiques", "Développement ou renforcement des compétences"]
}

# --- Appliquer multi-hot encoding ---
for col, labels in multi_choice_cols.items():
    df = multi_label_binarizer(df, col, labels)

# --- Liste des features mises à jour ---
features = [
    "Photoshop", "Creatif", "Domaine", "HTML", "TestLog", "ExpAvant",
    "Creatived", "Equipe", "Anglais", "NoteDip"
]

# Ajouter colonnes multi-hot générées
for col, labels in multi_choice_cols.items():
    features.extend([f"{col}_{label}" for label in labels])

# --- Cible ---
target = "Formation"

# --- Encodage label cible ---
le = LabelEncoder()
df["Formation_num"] = le.fit_transform(df[target])
os.makedirs("models", exist_ok=True)
joblib.dump(le, "models/label_encoder.pkl")

# --- Séparation X / y ---
X = df[features]
y = df["Formation_num"]

# --- Définir les colonnes catégoriques (Domaine uniquement) ---
cat_features = ["Domaine"]
num_features = [col for col in features if col != "Domaine"]

# Convertir les colonnes catégoriques en string (nécessaire pour OneHotEncoder)
X[cat_features] = X[cat_features].astype(str)

# --- Préprocesseur ---
preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="mean"), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])


X.head(5).to_excel("sample_encoded_before_training.xlsx", index=False)
y.head(5).to_excel("fore_training.xlsx", index=False)
print("✅ Échantillon encodé sauvegardé dans 'sample_encoded_before_training.xlsx'")
import pandas as pd

# Transformer toutes les données X (pas seulement les 5 premiers)
X_encoded = preprocessor.fit_transform(X)

# Récupérer noms colonnes OneHot
ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)

# Colonnes numériques
num_cols = num_features

# Construire DataFrame complet encodé (avec toutes les lignes)
df_encoded = pd.DataFrame(
    X_encoded.toarray() if hasattr(X_encoded, "toarray") else X_encoded, 
    columns = list(num_cols) + list(ohe_cols)
)

# Concaténer avec la colonne cible y (en gardant l'index aligné)
df_final = pd.concat([df_encoded.reset_index(drop=True), y.reset_index(drop=True)], axis=1)

# Sauvegarder dans un fichier Excel (toutes les lignes)
df_final.to_excel("bdd_complete_encoded.xlsx", index=False)

print("✅ Base complète encodée + cible sauvegardée dans 'bdd_complete_encoded.xlsx'")




# --- Modèles ---
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "GradientBoostingmeil": GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=3,
    loss="log_loss",
    criterion="friedman_mse",
    max_features=None,
    subsample=1.0,
    tol=0.0001,
    random_state=None  # ou mets `42` si tu veux garder un comportement déterministe
),

    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "Bagging": BaggingClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True, random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=500),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "LogisticRegressionmeilleur":LogisticRegression(C=1, class_weight='balanced', dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=500, multi_class="deprecated", n_jobs=None, penalty='l2', random_state =None, solver ='lbfgs', tol=0.0001, verbose=0, warm_start=False),
    "Ridge": RidgeClassifier(class_weight="balanced"),
}

# --- Stacking model ---
models["Stacking"] = StackingClassifier(
    estimators=[
        ("rf", models["RandomForest"]),
        ("gb", models["GradientBoosting"])
    ],
    final_estimator=LogisticRegression(class_weight="balanced")
)

# --- Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- Entraînement et sauvegarde ---
for name, clf in models.items():
    try:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipeline.fit(X_train, y_train)

        joblib.dump(pipeline, f"models/{name}.pkl")
        print(f"✅ Modèle {name} sauvegardé avec succès.")
    except Exception as e:
        print(f"❌ Erreur avec le modèle {name} : {e}")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Charger le pipeline du modèle logistic simple (si déjà sauvegardé)
# pipeline_logistic = joblib.load("models/LogisticRegression.pkl")

# Sinon, utiliser directement le pipeline entraîné
pipeline_logistic = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500, class_weight="balanced"))
])
pipeline_logistic.fit(X_train, y_train)

# Prédictions sur le test set
y_pred = pipeline_logistic.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

# Precision, Recall et F1 (macro pour moyenne sur toutes les classes)
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print(f"--- LogisticRegression sans SMOTE ---")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro): {rec:.4f}")
print(f"F1-score (macro): {f1:.4f}")

# Optionnel : rapport complet par classe
print("\nClassification report détaillé par classe :")
print(classification_report(y_test, y_pred, target_names=le.classes_))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# --- Préparation des features ---
X["Domaine"] = X["Domaine"].astype(str)
cat_features = ["Domaine"]
num_features = [col for col in features if col != "Domaine"]

preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="mean"), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# --- Logistic Regression avec class_weight ---
logreg_pipeline_weighted = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced"  # pondération automatique des classes
    ))
])

# --- GridSearchCV simple pour hyperparamètres ---
param_grid = {
    "classifier__C": [0.1, 1, 10],
    "classifier__penalty": ["l1", "l2"]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_logreg_weighted = GridSearchCV(
    logreg_pipeline_weighted,
    param_grid,
    cv=cv_strategy,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=2
)

gs_logreg_weighted.fit(X_train, y_train)

# --- Prédiction et sauvegarde ---
y_pred_weighted = gs_logreg_weighted.predict(X_test)
joblib.dump(gs_logreg_weighted.best_estimator_, "models/logistic_class_weighted.pkl")

# --- Affichage des scores ---
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

results_weighted = {
    "Accuracy": accuracy_score(y_test, y_pred_weighted),
    "Precision": precision_score(y_test, y_pred_weighted, average="weighted", zero_division=0),
    "Recall": recall_score(y_test, y_pred_weighted, average="weighted", zero_division=0),
    "F1": f1_score(y_test, y_pred_weighted, average="weighted", zero_division=0)
}

print("🎯 Meilleurs paramètres (Logistic avec class_weight) :")
print(gs_logreg_weighted.best_params_)

print("\n📊 Scores sur le test set :")
for metric, value in results_weighted.items():
    print(f"{metric}: {value:.4f}")
