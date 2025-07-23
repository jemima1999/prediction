import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- Chargement des données ---
df = pd.read_excel("bdd final.xlsx")

# --- Renommage des colonnes ---
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
    "Quelle a été votre motivation principale ou l'élément déclencheur pour la reconversion (expliqué en détail)":"Motivation",
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
    "Quelle formation avez vous suivie (filière/spécialité)": "Formation"
}
df.rename(columns=new_names, inplace=True)

# --- Features choisies ---
final_features = ["TestLog", "Photoshop", "HTML", "ExpAvant", "LangProg", "Motivation"]

# --- Encode target ---
target = "Formation"
le = LabelEncoder()
df["Formation_num"] = le.fit_transform(df[target])
y = df["Formation_num"]

# --- Traitement variables catégorielles ---
X = df[final_features]

if "LangProg" in X.columns and X["LangProg"].dtype == "object":
    X = pd.get_dummies(X, columns=["LangProg"], drop_first=True)

if "Motivation" in X.columns and X["Motivation"].dtype == "object":
    X = pd.get_dummies(X, columns=["Motivation"], drop_first=True)

# --- Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Random Forest ---
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", max_depth=5, min_samples_leaf=5)
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_test, y_test)

# --- Logistic Regression ---
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train, y_train)
log_score = log_model.score(X_test, y_test)

# --- Decision Tree ---
tree_model = DecisionTreeClassifier(random_state=42, class_weight="balanced", max_depth=5, min_samples_leaf=5)
tree_model.fit(X_train, y_train)
tree_score = tree_model.score(X_test, y_test)

# --- Affichage des résultats ---
print("=== Scores Test ===")
print(f"Random Forest : {rf_score:.2f}")
print(f"Logistic Regression : {log_score:.2f}")
print(f"Decision Tree : {tree_score:.2f}")

# --- Cross-validation (moyenne sur 5 folds) ---
rf_cv = cross_val_score(rf_model, X, y, cv=5).mean()
log_cv = cross_val_score(log_model, X, y, cv=5).mean()
tree_cv = cross_val_score(tree_model, X, y, cv=5).mean()

print("\n=== Cross-validation mean ===")
print(f"Random Forest : {rf_cv:.2f}")
print(f"Logistic Regression : {log_cv:.2f}")
print(f"Decision Tree : {tree_cv:.2f}")

# --- Classification report détaillé pour Random Forest (exemple) ---
y_pred_rf = rf_model.predict(X_test)
target_names_present = le.inverse_transform(sorted(y_test.unique()))

print("\n=== Classification report Random Forest ===")
print(classification_report(y_test, y_pred_rf, labels=sorted(y_test.unique()), target_names=target_names_present))

print("\nMatrice de confusion Random Forest :")
print(confusion_matrix(y_test, y_pred_rf))
