
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from api.explain import ShapExplainer

# Chemins
ARTIFACTS = Path("artifacts")

# 1. Charger le pipeline et les données
pipeline = joblib.load(
    ARTIFACTS / "xgboost_pipeline.joblib"
)

df = pd.read_csv(ARTIFACTS / "test_set.csv")

# 2. Conserver les 36 variables explicatives
features = list(pipeline.feature_names_in_)
X_test = df[features].copy()

# 3. Initialiser le moteur SHAP existant
engine = ShapExplainer(pipeline)

# 4. Transformer les 885 observations
X_transformed = engine.prep.transform(X_test)

# 5. Calculer les contributions SHAP
print("Calcul des valeurs SHAP...")

shap_values = np.asarray(
    engine.explainer.shap_values(X_transformed)
)

# 6. Regrouper les contributions par variable
shap_grouped = shap_values @ engine.M

print("Dimensions :", shap_grouped.shape)

# 7. Vérifier la conservation des contributions
difference = np.max(
    np.abs(
        shap_values.sum(axis=1)
        - shap_grouped.sum(axis=1)
    )
)

print("Erreur maximale de regroupement :", difference)

assert difference < 1e-4

# 8. Vérifier la reconstruction des prédictions
base_value = float(
    np.ravel(engine.explainer.expected_value)[0]
)

raw_predictions = base_value + shap_grouped.sum(axis=1)

probabilities_reconstructed = (
    1 / (1 + np.exp(-raw_predictions))
)

probabilities_model = pipeline.predict_proba(X_test)[:, 1]

prediction_error = np.max(
    np.abs(
        probabilities_reconstructed - probabilities_model
    )
)

print("Erreur maximale sur les probabilités :", prediction_error)

assert prediction_error < 1e-4

# 9. Calculer l'importance globale
importance = pd.Series(
    np.abs(shap_grouped).mean(axis=0),
    index=features,
    name="importance_shap"
).sort_values(ascending=False)

print("\nTop 10 des variables :")
print(importance.head(10).round(4))

# 10. Enregistrer les résultats
importance.to_csv(
    ARTIFACTS / "shap_global_importance.csv",
    index_label="variable"
)

np.savez_compressed(
    ARTIFACTS / "shap_global.npz",
    values=shap_grouped,
    base_value=np.array(base_value),
    feature_names=np.array(features)
)

print("\nFichiers enregistrés :")
print("- artifacts/shap_global_importance.csv")
print("- artifacts/shap_global.npz")
