
import joblib
import numpy as np
import pandas as pd
import shap

# 1. Charger les données et le modèle
pipeline = joblib.load("artifacts/xgboost_pipeline.joblib")
df = pd.read_csv("artifacts/test_set.csv")

features = list(pipeline.feature_names_in_)
student = df[features].iloc[[0]]

prep = pipeline.named_steps["prep"]
model = pipeline.named_steps["clf"]

# 2. Identifier les colonnes transformées
encoded_names = list(prep.get_feature_names_out())

# Récupérer les variables catégorielles depuis le préprocesseur
# Identifier le transformeur catégoriel par son nom
categorical_cols = list(
    next(
        columns
        for name, transformer, columns in prep.transformers_
        if name == "cat"
    )
)

# Trier du nom le plus long au plus court
categorical_cols = sorted(
    categorical_cols,
    key=len,
    reverse=True
)

print("Variables catégorielles :", len(categorical_cols))
print(categorical_cols)
def original_feature(encoded_name):
    # Exemple : "cat__Course_33" -> "Course"
    kind, name = encoded_name.split("__", 1)

    if kind == "cat":
        for col in categorical_cols:
            if name.startswith(col + "_"):
                return col

        raise ValueError(f"Catégorie inconnue : {encoded_name}")

    if name not in features:
        raise ValueError(f"Variable inconnue : {encoded_name}")

    return name

# 3. Construire la matrice de regroupement
M = np.zeros((len(encoded_names), len(features)))

for i, name in enumerate(encoded_names):
    origin = original_feature(name)
    M[i, features.index(origin)] = 1

assert np.all(M.sum(axis=1) == 1)

# 4. Calculer SHAP
X_transformed = prep.transform(student)

explainer = shap.TreeExplainer(model)
shap_values = np.asarray(explainer.shap_values(X_transformed))

# 5. Regrouper les contributions
shap_grouped = shap_values @ M

# 6. Vérifier la conservation
print("Variables transformées :", len(encoded_names))
print("Variables originales :", len(features))

print("Somme SHAP avant :", shap_values.sum())
print("Somme SHAP après :", shap_grouped.sum())

print(
    "Erreur de regroupement :",
    abs(shap_values.sum() - shap_grouped.sum())
)

# 7. Afficher les principales contributions
importance = pd.Series(
    shap_grouped[0],
    index=features
)

importance = importance.reindex(
    importance.abs().sort_values(ascending=False).index
)

print("\nTop 10 des contributions pour cet étudiant :")
print(importance.head(10).round(4))
