import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder    # ✅ AJOUTER CETTE LIGNE
import matplotlib.pyplot as plt
import seaborn as sns


# 🟢 Charger ton dataset
df = pd.read_excel('bdd.xlsx')  # Mets ton fichier ici

# ✅ Nettoyage simple
df['Genre'] = df['Genre'].map({'M': 1, 'F': 0})
df['Aviez vous un ordinateur personnel durant la formation ?'] = df['Aviez vous un ordinateur personnel durant la formation ?'].map({'Oui': 1, 'Non': 0})
df['Aviez vous accès à une Connexion Internet stable chez vous à la maison ?'] = df['Aviez vous accès à une Connexion Internet stable chez vous à la maison ?'].map({'Oui': 1, 'Non': 0})

# ✅ Encodage one-hot
vars_onehot = ['Pays de provenance', 'Domaine d’études avant la reconversion', 'Niveau d’études le plus élevé avant la reconversion']
df = pd.get_dummies(df, columns=vars_onehot)

# ✅ Encodage colonnes à choix multiples
def split_multi(x):
    if pd.isnull(x):
        return []
    return [v.strip() for v in x.split(',')]

multi_cols = [
    'Langages de programmation connus avant la formation/reconversion',
    'Outils de design maîtrisés avant la formation/reconversion',
    'Compétences bureautiques avant la formation/reconversion',
    'Compétences transversales avant la formation/reconversion'
]

for col in multi_cols:
    mlb = MultiLabelBinarizer()
    df[col] = df[col].fillna('')
    df[col+'_list'] = df[col].apply(split_multi)
    dummies = pd.DataFrame(mlb.fit_transform(df[col+'_list']), columns=[f"{col}_{c}" for c in mlb.classes_])
    df = pd.concat([df, dummies], axis=1)
    df = df.drop([col, col+'_list'], axis=1)

# ✅ Encodage variable cible
df['Formation_num'] = df['Quelle formation avez vous suivie (filière/spécialité)'].astype('category').cat.codes

# ✅ Vérifier le mapping Formation_num
print("\n✅ Mapping formation vs code :")
print(df[['Quelle formation avez vous suivie (filière/spécialité)', 'Formation_num']].drop_duplicates())

# ✅ Sélection des colonnes numériques
num_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
if 'Formation_num' not in num_cols:
    num_cols.append('Formation_num')

print("\n✅ Colonnes numériques avant matrice :")
print(num_cols)

print("\n✅ Colonnes du DataFrame original :")
print(df.columns)

print("\n✅ Aperçu des valeurs uniques de Formation_num :")
print(df['Formation_num'].unique())

# ✅ Matrice de corrélation
corr_matrix = df[num_cols].corr()

# ✅ Renommer colonnes (pour affichage heatmap)
new_names = {
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
    "Outils de design maîtrisés avant la formation/reconversion": "Design",
    "Compétences bureautiques avant la formation/reconversion": "Bureautique",
    "Compétences transversales avant la formation/reconversion": "Transv",
    "Aviez vous un ordinateur personnel durant la formation ?": "Ordi",
    "Aviez vous accès à une Connexion Internet stable chez vous à la maison ?": "Internet",
    "Êtes vous plutôt créatif ? Préférence logique vs. créatif (1=Très créatif, 5=Très logique)": "Creatif",
    "Préférez vous travailler en équipe ? Travail en équipe vs solo (1=Solo, 5=Équipe)": "Equipe",
    "Quel a été votre Résultat au test de logique (0-10) ** Si vous avez eu accès aux résultats": "TestLog",
    "Quel a été votre Résultat au test de créativité  (0-10) ** Si vous avez eu accès aux résultats": "TestCrea",
    "Quel était votre niveau de maîtrise du logiciel photoshop avant la formation ? Auto-évaluation Photoshop (1 à 5)": "Photoshop",
    "Quel était votre niveau de maîtrise de HTML avant la formation ? Auto-évaluation HTML/CSS (1 à 5)": "HTML",
    "Niveau de langue étrangère - anglais - avant la formation (1 élémentaire - 5 courant)": "Anglais",
    "Quelle formation avez vous suivie (filière/spécialité)": "Formation"
}

corr_matrix = corr_matrix.rename(columns=new_names, index=new_names)

# ✅ Extraire corrélations avec la cible
target_corr = corr_matrix['Formation_num'].drop('Formation_num').sort_values(ascending=False)

# ✅ Heatmap
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("Matrice de corrélation complète", fontsize=15)
plt.tight_layout()
plt.show()

# ✅ Top 10 des variables corrélées à la cible
top10 = target_corr.head(30)

plt.figure(figsize=(10, 6))
sns.barplot(x=top10.values, y=top10.index, palette='viridis')
plt.title("Top 10 variables les plus corrélées à la formation cible", fontsize=14)
plt.xlabel("Coefficient de corrélation")
plt.ylabel("Variables")
plt.tight_layout()
plt.show()

print("\n✅ Top 10 variables les plus corrélées à la cible :")
print(top10)
# 💡 Encoder la variable cible
le = LabelEncoder()
df["Formation_num"] = le.fit_transform(df["Quelle formation avez vous suivie (filière/spécialité)"])

# 🎯 Définir X et y
X = df.drop(columns=["Quelle formation avez vous suivie (filière/spécialité)", "Formation_num"])  # on retire la cible
y = df["Formation_num"]

# 🔎 Pour s'assurer que toutes les colonnes sont numériques (au besoin, encoder avant)
X = pd.get_dummies(X, drop_first=True)

# 💥 Remplacer les NaN par 0 (ou autre stratégie de ton choix)
X = X.fillna(0)

# ✅ Créer et entraîner le modèle Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 🔥 Extraire les importances
importances = rf.feature_importances_
features = X.columns

# 🟢 Créer un DataFrame pour trier et afficher
feat_importances = pd.DataFrame({"Feature": features, "Importance": importances})
feat_importances = feat_importances.sort_values(by="Importance", ascending=False)

print(feat_importances)

# 🎨 Visualiser les 20 variables les plus importantes
plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=feat_importances.head(20))
plt.title("Top 20 des variables importantes (Random Forest)")
plt.tight_layout()
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ✅ Charger ton DataFrame déjà nettoyé
# df = pd.read_csv("ton_fichier.csv")  # si depuis un fichier

# ✅ Dictionnaire de renommage
new_names = {
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
    "Outils de design maîtrisés avant la formation/reconversion": "Design",
    "Compétences bureautiques avant la formation/reconversion": "Bureautique",
    "Compétences transversales avant la formation/reconversion": "Transv",
    "Aviez vous un ordinateur personnel durant la formation ?": "Ordi",
    "Aviez vous accès à une Connexion Internet stable chez vous à la maison ?": "Internet",
    "Êtes vous plutôt créatif ? Préférence logique vs. créatif (1=Très créatif, 5=Très logique)": "Creatif",
    "Préférez vous travailler en équipe ? Travail en équipe vs solo (1=Solo, 5=Équipe)": "Equipe",
    "Quel a été votre Résultat au test de logique (0-10) ** Si vous avez eu accès aux résultats": "TestLog",
    "Quel a été votre Résultat au test de créativité  (0-10) ** Si vous avez eu accès aux résultats": "TestCrea",
    "Quel était votre niveau de maîtrise du logiciel photoshop avant la formation ? Auto-évaluation Photoshop (1 à 5)": "Photoshop",
    "Quel était votre niveau de maîtrise de HTML avant la formation ? Auto-évaluation HTML/CSS (1 à 5)": "HTML",
    "Niveau de langue étrangère - anglais - avant la formation (1 élémentaire - 5 courant)": "Anglais",
    "Quelle formation avez vous suivie (filière/spécialité)": "Formation"
}

# ✅ Renommer
df.rename(columns=new_names, inplace=True)

# ✅ Variables finales (reprendre avec les nouveaux noms)
final_features = ["TestLog", "Photoshop", "ExpAvant", "LangProg", "HTML"]

# ✅ Variable cible
target = "Formation"

# ✅ Encodage de la cible
le = LabelEncoder()
df["Formation_num"] = le.fit_transform(df[target])

# ✅ Sélection des features
X = df[final_features]
y = df["Formation_num"]

# ✅ Encodage éventuel si une feature est textuelle (ex : LangProg)
if X["LangProg"].dtype == "object":
    X = pd.get_dummies(X, columns=["LangProg"], drop_first=True)

# ✅ Vérifier colonnes numériques avant la matrice
num_cols = list(X.select_dtypes(include=['int64', 'float64']).columns)
if "Formation_num" not in num_cols:
    num_cols.append("Formation_num")
print("Colonnes numériques finales :")
print(num_cols)

print("Colonnes du DataFrame original :")
print(df.columns)

print("Aperçu Formation_num :")
print(df[['Formation', 'Formation_num']].head())

# ✅ Séparation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Modèle
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ✅ Scores
print("Score train :", model.score(X_train, y_train))
print("Score test :", model.score(X_test, y_test))

y_pred = model.predict(X_test)

print("\nClassification report :")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

# ✅ Importances
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nImportances des variables :")
print(importances)
