import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('bdd final.xlsx')

df['Genre'] = df['Genre'].map({'M': 1, 'F': 0})
df['Aviez vous un ordinateur personnel durant la formation ?'] = df['Aviez vous un ordinateur personnel durant la formation ?'].map({'Oui': 1, 'Non': 0})
df['Aviez vous accès à une Connexion Internet stable chez vous à la maison ?'] = df['Aviez vous accès à une Connexion Internet stable chez vous à la maison ?'].map({'Oui': 1, 'Non': 0})

vars_onehot = ['Pays de provenance', 'Domaine d’études avant la reconversion', 'Niveau d’études le plus élevé avant la reconversion']
df = pd.get_dummies(df, columns=vars_onehot)

multi_cols = [
    'Langages de programmation connus avant la formation/reconversion',
    'Outils de design maîtrisés avant la formation/reconversion',
    'Compétences bureautiques avant la formation/reconversion',
    'Compétences transversales avant la formation/reconversion',
    "Quelle a été votre motivation principale ou l'élément déclencheur pour la reconversion (expliqué en détail)"

]

def split_multi(x):
    if pd.isnull(x):
        return []
    return [v.strip() for v in x.split(',')]

for col in multi_cols:
    mlb = MultiLabelBinarizer()
    df[col] = df[col].fillna('')
    df[col + '_list'] = df[col].apply(split_multi)
    dummies = pd.DataFrame(mlb.fit_transform(df[col + '_list']), columns=[f"{col}_{c}" for c in mlb.classes_])
    df = pd.concat([df, dummies], axis=1)
    df = df.drop([col, col + '_list'], axis=1)

le = LabelEncoder()
df['Formation_num'] = le.fit_transform(df['Quelle formation avez vous suivie (filière/spécialité)'])

X = df.drop(columns=['Quelle formation avez vous suivie (filière/spécialité)', 'Formation_num'])
y = df['Formation_num']

X = pd.get_dummies(X, drop_first=True)
X = X.fillna(0)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feat_importances = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
feat_importances = feat_importances.sort_values(by="Importance", ascending=False)

print(feat_importances.head(30))

plt.figure(figsize=(12, 10))
sns.barplot(x="Importance", y="Feature", data=feat_importances.head(30))
plt.title("Top 30 variables importantes (Random Forest)")

plt.subplots_adjust(left=0.60)
plt.show()
