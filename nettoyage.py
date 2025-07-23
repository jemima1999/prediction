import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
df = pd.read_excel("data.xlsx")


def nettoyer_age(x):
    if pd.isna(x):
        return None
    x = str(x)
    res = re.findall(r'\d+', x)
    if res:
        return int(res[0])
    else:
        return None

df['Âge_nettoye'] = df['Âge'].apply(nettoyer_age)
df['Âge'] = df['Âge_nettoye']
df.drop(columns=['Âge_nettoye'], inplace=True)

print(df['Âge'].unique())


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Créer une liste de stopwords français
stopwords = set(STOPWORDS)
stopwords.update([
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "le", "la", "les", "un", "une", "des", "du", "de", "d", 
    "en", "et", "à", "au", "aux", "pour", "par", "sur", "avec", "ce", "cet", "cette", "ces", "dans", "qui", "que", "quoi", "dont", "où", "comme","Être","dans","en","est","J'ai","j'ai","suis","plus",
])

# Concaténer tout le texte
text = " ".join(str(mot) for mot in df["Quelle a été votre motivation principale ou l'élément déclencheur pour la reconversion (expliqué en détail)"].dropna())

# Créer le wordcloud
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    stopwords=stopwords,
    collocations=False  # très important pour éviter les répétitions automatiques
).generate(text)

# Afficher
plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Nuage de mots nettoyé - Motivation principale")
plt.show()

"""

text = " ".join(str(mot) for mot in df["Quelle a été votre motivation principale ou l'élément déclencheur pour la reconversion (expliqué en détail)"].dropna())

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Nuage de mots - Motivation principale")
plt.show()

"""