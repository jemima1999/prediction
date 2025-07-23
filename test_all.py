import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Modèles de régression
from sklearn.linear_model import BayesianRidge, LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Chargement données et renommage colonnes (adapter selon ta base)
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
    # Remplace "target" par ta variable cible continue (exemple : "Note finale", "Score", etc.)
    "Note finale (exemple cible)": "target"  
}

df.rename(columns=new_names, inplace=True)

# Liste des features choisies (adapter selon tes variables)
final_features = ["TestLog", "Photoshop", "HTML", "ExpAvant", "LangProg", "Motivation"]

# S’assurer que la cible est bien dans le dataframe
if "target" not in df.columns:
    raise ValueError("La variable cible 'target' n'est pas présente dans le dataframe.")

# Retirer les lignes sans cible (NaN)
df = df.dropna(subset=["target"])

# Préparation X et y
X = df[final_features]
y = df["target"]

# Encodage des variables catégorielles
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionnaire des modèles (avec pipeline scaler pour certains)
models = {
    "Bayesian Ridge": BayesianRidge(),
    "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsRegressor()),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42, eval_metric="rmse"),
    "Support Vector Regression": make_pipeline(StandardScaler(), SVR()),
    "Elastic Net": ElasticNet(random_state=42),
    "Multilayer Perceptron": make_pipeline(StandardScaler(), MLPRegressor(random_state=42, max_iter=1000)),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(random_state=42),
    "Bagging Regressor": BaggingRegressor(random_state=42),
    "Stacking Ensemble": StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(random_state=42)),
            ("svr", make_pipeline(StandardScaler(), SVR())),
            ("br", BayesianRidge()),
        ],
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1,
    )
}

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"=== {name} ===")
    print(f"R2 score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}\n")

# Boucle d’évaluation
for name, model in models.items():
    evaluate_model(name, model, X_train, y_train, X_test, y_test)
