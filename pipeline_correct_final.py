import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

# --- Fonction multi-hot encoding ---
def multi_label_binarizer(df, column_name, possible_labels):
    df[column_name] = df[column_name].fillna("")
    for label in possible_labels:
        df[f"{column_name}_{label}"] = df[column_name].apply(
            lambda x: int(label in [v.strip() for v in x.split(',')])
        )
    df.drop(columns=[column_name], inplace=True)
    return df

# --- Chargement et préparation des données ---
df = pd.read_excel("bdd final.xlsx")

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

multi_choice_cols = {
    "LangProg": ["Python", "JavaScript", "Java", "HTML / CSS", "Aucun (Néant)"],
    "Bureautique": ["Microsoft Word", "Microsoft Excel", "Google Sheets", "Google Slides", "Powerpoint", "Jira", "Aucun (Néant)"],
    "Design": ["Adobe Photoshop", "Canva", "Adobe Illustrator", "Aucun (Néant)", "Autre"],
    "Transv": ["Gestion de projet", "Rédaction documentaire", "Gestion du stress", "Autonomie", "Proactivité", "Aucune (Néant)", "Autre"],
    "Motivation": ["Passion et intérêt personnel", "Motivations professionnelles", "Motivations économiques", "Développement ou renforcement des compétences"]
}

for col, labels in multi_choice_cols.items():
    df = multi_label_binarizer(df, col, labels)

features = ["Photoshop", "Creatif", "Domaine", "HTML", "TestLog", "ExpAvant", "Creatived", "Equipe", "Anglais", "NoteDip"]
for col, labels in multi_choice_cols.items():
    features.extend([f"{col}_{label}" for label in labels])

target = "Formation"
le = LabelEncoder()
df["Formation_num"] = le.fit_transform(df[target])
os.makedirs("models", exist_ok=True)
joblib.dump(le, "models/label_encoder.pkl")

X = df[features].copy()  # <-- .copy() pour éviter SettingWithCopyWarning
X["Domaine"] = X["Domaine"].astype(str)
y = df["Formation_num"]

cat_features = ["Domaine"]
num_features = [col for col in features if col != "Domaine"]

preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="mean"), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# --- Déterminer k_neighbors pour SMOTE ---
min_class_count = min(Counter(y_train).values())
smote_k = max(1, min(5, min_class_count-1))  # max 5 pour sécurité
print("⚠️ k_neighbors pour SMOTE =", smote_k)

# --- Modèles ---
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "Bagging": BaggingClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True, random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=500),
    "NaiveBayes": GaussianNB(),
    "Ridge": RidgeClassifier(),
}

results = {}

# --- Entraînement avec SMOTE pour tous les modèles ---
for name, clf in models.items():
    try:
        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42, k_neighbors=smote_k)),
            ("classifier", clf)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        joblib.dump(pipeline, f"models/{name}_smote.pkl")
        
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }
        print(f"✅ Modèle {name} entraîné avec SMOTE et sauvegardé.")
    except Exception as e:
        print(f"❌ Erreur avec le modèle {name}: {e}")

# --- Logistic Regression avec GridSearchCV sécurisée ---
safe_smote = SMOTE(random_state=42, k_neighbors=1)  # k=1 pour éviter erreurs folds

logreg_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", safe_smote),
    ("classifier", LogisticRegression(max_iter=1000, solver="liblinear"))
])

param_grid_logreg = {
    "classifier__C": [0.1, 1, 10],
    "classifier__penalty": ["l1", "l2"]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs_logreg = GridSearchCV(
    logreg_pipeline,
    param_grid_logreg,
    cv=cv_strategy,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=2,
    error_score='raise'
)

gs_logreg.fit(X_train, y_train)
y_pred_logreg = gs_logreg.predict(X_test)

# --- Refit final avec SMOTE k_neighbors correct ---
best_pipeline = gs_logreg.best_estimator_
best_pipeline.set_params(smote=SMOTE(random_state=42, k_neighbors=smote_k))
best_pipeline.fit(X_train, y_train)
joblib.dump(best_pipeline, "models/logistic_ajuste_smote.pkl")

results["Logistic_ajuste"] = {
    "Accuracy": accuracy_score(y_test, y_pred_logreg),
    "Precision": precision_score(y_test, y_pred_logreg, average="weighted", zero_division=0),
    "Recall": recall_score(y_test, y_pred_logreg, average="weighted", zero_division=0),
    "F1": f1_score(y_test, y_pred_logreg, average="weighted", zero_division=0)
}

print("\n🎯 Meilleurs paramètres (logistic_ajuste):")
print(gs_logreg.best_params_)

print("\n📊 Scores comparatifs sur le test set :")
for model_name, metrics in results.items():
    print(f"\n--- {model_name} ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

print("\n📁 Tous les modèles sauvegardés dans 'models/'")
