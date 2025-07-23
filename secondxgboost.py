import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

df = pd.read_excel("bdd final.xlsx")

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

final_features = ["TestLog", "Photoshop", "ExpAvant", "LangProg", "HTML"]
target = "Formation"

le = LabelEncoder()
df["Formation_num"] = le.fit_transform(df[target])

X = df[final_features]
y = df["Formation_num"]
print(df["Formation_num"].value_counts())


if X["LangProg"].dtype == "object":
    X = pd.get_dummies(X, columns=["LangProg"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
weights_dict = dict(zip(classes, class_weights))
sample_weights = np.array([weights_dict[label] for label in y_train])

model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(classes),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    n_estimators=200
)

print(df["Formation_num"].value_counts())

model.fit(X_train, y_train, sample_weight=sample_weights)

print("Score train :", model.score(X_train, y_train))
print("Score test :", model.score(X_test, y_test))

y_pred = model.predict(X_test)

labels_present = np.unique(y_test)
target_names_present = le.inverse_transform(labels_present)

print("\nClassification report :")
print(classification_report(y_test, y_pred, labels=labels_present, target_names=target_names_present))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))
