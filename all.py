import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, balanced_accuracy_score
)
# 📦 Modèles
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, StackingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# 📥 Chargement
df = pd.read_excel("bdd final.xlsx")

# 🎯 Renommage
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
    "Êtes vous plutôt créatif ? Préférence logique vs. créatif (1=Très créatif, 5=Très logique)": "Creatif",
    "Préférez vous travailler en équipe ? Travail en équipe vs solo (1=Solo, 5=Équipe)": "Equipe",
    "Quel a été votre Résultat au test de logique (0-10) ** Si vous avez eu accès aux résultats": "TestLog",
    "Quel était votre niveau de maîtrise du logiciel photoshop avant la formation ? Auto-évaluation Photoshop (1 à 5)": "Photoshop",
    "Quel était votre niveau de maîtrise de HTML avant la formation ? Auto-évaluation HTML/CSS (1 à 5)": "HTML",
    "Niveau de langue étrangère - anglais - avant la formation (1 élémentaire - 5 courant)": "Anglais",
    "Quelle formation avez vous suivie (filière/spécialité)": "Formation"
}, inplace=True)

# 🧠 Features (les mêmes que ton code initial)
features = ["Photoshop","Creatif","Domaine","HTML","TestLog","ExpAvant","Transv",
            "AnneeDip","Design","Bureautique","Motivation","Anglais","LangProg",
            "Equipe","NoteDip"]

features = [f for f in features if f in df.columns]

# 🎯 Cible
target = "Formation"
le = LabelEncoder()
df["Formation_num"] = le.fit_transform(df[target])

# 🛠️ Encodage
X = df[features]
X = pd.get_dummies(X, drop_first=True)
y = df["Formation_num"]

# 📊 Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 🔍 Modèles avec mêmes hyperparamètres
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
    "Bagging": BaggingClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True, random_state=42),
    "MLP": MLPClassifier(random_state=42, max_iter=500),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "Ridge": RidgeClassifier(class_weight="balanced"),
}

# Stacking
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
    ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42))
]
stacking_model = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(class_weight="balanced"))
models["Stacking"] = stacking_model

# 📊 Résultats
results = []

print("====== ÉVALUATION DES MODÈLES ======")
for name, model in models.items():
    try:
        model.fit(X_train, y_train)

        # Scores train/test
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Prédictions
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        # Sauvegarde
        results.append({
            "Modèle": name,
            "Train Score": train_score,
            "Test Score": test_score,
            "Accuracy": acc,
            "F1 Macro": f1_macro,
            "Balanced Acc": bal_acc
        })

        # 📄 Rapport détaillé
        print(f"\n📌 Modèle : {name}")
        print(f"✅ Train Score : {train_score:.3f}")
        print(f"✅ Test Score : {test_score:.3f}")
        print(f"✅ Accuracy : {acc:.3f}")
        print(f"✅ F1 Macro : {f1_macro:.3f}")
        print(f"✅ Balanced Acc : {bal_acc:.3f}")
        print("📋 Classification Report :")
        print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
        print("📊 Matrice de confusion :")
        print(confusion_matrix(y_test, y_pred))

    except Exception as e:
        print(f"\n❌ Erreur avec le modèle {name} : {e}")

# 📄 DataFrame récap
df_results = pd.DataFrame(results)
print("\n====== 📊 RÉCAPITULATIF FINAL ======")
print(df_results.sort_values(by="Test Score", ascending=False))
import os
import joblib

# 📁 Création du dossier de sauvegarde si nécessaire
os.makedirs("models", exist_ok=True)

# 🔁 Entraînement et sauvegarde des modèles
for name, model in models.items():
    try:
        model.fit(X_train, y_train)

        # Enregistrement du modèle dans le dossier "models/"
        model_filename = f"models/{name.replace(' ', '_')}.pkl"
        joblib.dump(model, model_filename)
        print(f"✅ Modèle {name} enregistré sous : {model_filename}")

        # Évaluation (facultatif mais gardé pour affichage)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        results.append({
            "Modèle": name,
            "Train Score": train_score,
            "Test Score": test_score,
            "Accuracy": acc,
            "F1 Macro": f1_macro,
            "Balanced Acc": bal_acc
        })

    except Exception as e:
        print(f"❌ Erreur avec le modèle {name} : {e}")
