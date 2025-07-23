import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_excel('bdd final.xlsx')

# Transformations binaires
df['Genre'] = df['Genre'].map({'M': 1, 'F': 0})
df['Aviez vous un ordinateur personnel durant la formation ?'] = df['Aviez vous un ordinateur personnel durant la formation ?'].map({'Oui': 1, 'Non': 0})
df['Aviez vous accès à une Connexion Internet stable chez vous à la maison ?'] = df['Aviez vous accès à une Connexion Internet stable chez vous à la maison ?'].map({'Oui': 1, 'Non': 0})

# One-hot encoding pour certaines colonnes
vars_onehot = ['Pays de provenance', 'Domaine d’études avant la reconversion', 'Niveau d’études le plus élevé avant la reconversion']
df = pd.get_dummies(df, columns=vars_onehot)

def split_multi(x):
    if pd.isnull(x):
        return []
    return [v.strip() for v in x.split(',')]

multi_cols = [
    'Langages de programmation connus avant la formation/reconversion',
    'Outils de design maîtrisés avant la formation/reconversion',
    'Compétences bureautiques avant la formation/reconversion',
    'Compétences transversales avant la formation/reconversion',
    "Quelle a été votre motivation principale ou l'élément déclencheur pour la reconversion (expliqué en détail)"

]

for col in multi_cols:
    mlb = MultiLabelBinarizer()
    df[col] = df[col].fillna('')
    df[col + '_list'] = df[col].apply(split_multi)
    dummies = pd.DataFrame(mlb.fit_transform(df[col + '_list']), columns=[f"{col}_{c}" for c in mlb.classes_])
    df = pd.concat([df, dummies], axis=1)
    df = df.drop([col, col + '_list'], axis=1)


df['Formation_num'] = df['Quelle formation avez vous suivie (filière/spécialité)'].astype('category').cat.codes

print(df["Formation_num"].value_counts())
print(df['Quelle formation avez vous suivie (filière/spécialité)'].value_counts())

# Sélection des colonnes numériques
num_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
if 'Formation_num' not in num_cols:
    num_cols.append('Formation_num')

# Corrélation
corr_matrix = df[num_cols].corr()
target_corr = corr_matrix['Formation_num'].drop('Formation_num').sort_values(ascending=False)

# Heatmap générale
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))  # définit la taille directement AVANT

sns.heatmap(corr_matrix, cmap='coolwarm', center=0)

plt.title("Matrice de corrélation complète ")

# Ajuster les marges pour serrer à droite
plt.tight_layout(rect=[0, 0, 0.95, 1])
#           ^  ^  ^  ^
#  left, bottom, right, top (ici on met right=0.95 pour rapprocher du bord droit)

plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

# Corr_matrix déjà calculée
# corr_matrix = df[num_cols].corr()

target_corr = corr_matrix['Formation_num'].drop('Formation_num').sort_values(ascending=False)
corr_df = target_corr.reset_index()
corr_df.columns = ['Variable', 'Corrélation avec Formation_num']

# Wrap dans un string final (pour un affichage "textuel" sur PDF)
wrap_width = 50
corr_df['Variable'] = corr_df['Variable'].apply(lambda x: "".join(textwrap.wrap(x, wrap_width)))

# Convertir le dataframe entier en string bien formaté
table_string = corr_df.to_string(index=False)

# Créer PDF
with PdfPages("corr_table.pdf") as pdf:
    fig, ax = plt.subplots(figsize=(8.5, len(corr_df) * 0.4))
    ax.axis('off')

    # Ajouter texte
    ax.text(0, 1, table_string, fontsize=9, ha='left', va='top', family='monospace')
    plt.title("Corrélations avec Formation_num", fontsize=14, pad=20)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print("✅ PDF créé proprement : 'corr_table.pdf' prêt !")



# Top 30
top30 = target_corr.head(30)
plt.figure(figsize=(10, 12))

sns.barplot(x=top30.values, y=top30.index, palette='viridis')
plt.title("Top 30 variables les plus corrélées à la cible")
plt.xlabel("Coefficient de corrélation")
plt.ylabel("Variables")

plt.subplots_adjust(left=0.60)
plt.show()


print("\nTop 30 variables les plus corrélées :")
print(top30)
