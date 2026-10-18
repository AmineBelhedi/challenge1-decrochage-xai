"""API d'interprétation - Challenge 1 : décrochage étudiant."""
import json
from pathlib import Path
from functools import lru_cache
from api.explain import ShapExplainer
import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

ART = Path(__file__).resolve().parent.parent / "artifacts"

meta = json.load(open(ART / "meta.json", encoding="utf-8"))
FEATURES = meta["features"]
INT_COLS = meta["categorical"] + meta["binary"]
DEFAULT_THRESHOLD = meta["default_threshold"]

MODEL_FILES = {
    "xgboost": "xgboost",
    "logreg": "logreg",
    "random_forest": "random_forest",
    "knn": "knn",
    "decision_tree": "decision_tree",
}
MODELS = {nom: joblib.load(ART / f"{fichier}_pipeline.joblib")
          for nom, fichier in MODEL_FILES.items()}

test_df = pd.read_csv(ART / "test_set.csv")
X_test = test_df[FEATURES]
y_test = test_df[meta["target"]]

EXAMPLE = {
    "modele": "xgboost",
    "seuil": DEFAULT_THRESHOLD,
    "etudiant": {k: float(v) for k, v in X_test.iloc[0].items()},
}

app = FastAPI(
    title="API de prédiction du décrochage étudiant",
    description=("Sert les 5 modèles entraînés (XGBoost, régression logistique, Random Forest, "
                 "k-NN, arbre de décision) : prédiction individuelle et métriques. "
                 "Projet Challenge 1, Data Mining, ENSI."),
    version="1.0.0",
)


class PredictRequest(BaseModel):
    modele: str = Field("xgboost", description=f"Modèle à utiliser : {list(MODEL_FILES)}")
    seuil: float = Field(DEFAULT_THRESHOLD, ge=0, le=1,
                         description="Seuil de décision : alerte si probabilité >= seuil")
    etudiant: dict[str, float] = Field(..., description="Les 36 variables de l'étudiant")


def get_model(name: str):
    if name not in MODELS:
        raise HTTPException(status_code=404,
                            detail=f"Modèle inconnu : '{name}'. Choix : {list(MODELS)}")
    return MODELS[name]


def to_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURES].copy()
    out[INT_COLS] = out[INT_COLS].astype(int)
    return out

@lru_cache(maxsize=1)
def get_shap_engine():
    """Initialiser SHAP une seule fois, au premier appel."""
    return ShapExplainer(MODELS["xgboost"])

@app.get("/health", summary="Vérifier que l'API fonctionne")
def health():
    return {"status": "ok", "models": list(MODELS), "n_test": len(test_df)}


@app.get("/models", summary="Lister les modèles disponibles")
def list_models():
    return {"models": list(MODELS), "default_threshold": DEFAULT_THRESHOLD}


@app.get("/metrics", summary="Métriques d'un modèle sur le jeu de test")
def metrics(modele: str = "xgboost",
            seuil: float = Query(DEFAULT_THRESHOLD, ge=0, le=1)):
    model = get_model(modele)
    proba = model.predict_proba(to_frame(X_test))[:, 1]
    pred = (proba >= seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "modele": modele,
        "seuil": seuil,
        "auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "rappel": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "matrice_confusion": {"vrais_negatifs": int(tn), "faux_positifs": int(fp),
                              "faux_negatifs": int(fn), "vrais_positifs": int(tp)},
    }


@app.post("/predict", summary="Prédire le risque de décrochage d'un étudiant")
def predict(req: PredictRequest = Body(
        ..., openapi_examples={"etudiant_du_jeu_de_test": {
            "summary": "Un étudiant réel du jeu de test", "value": EXAMPLE}})):
    model = get_model(req.modele)
    manquantes = [c for c in FEATURES if c not in req.etudiant]
    if manquantes:
        raise HTTPException(status_code=422,
                            detail=f"Variables manquantes : {manquantes}")
    row = pd.DataFrame([req.etudiant])
    proba = float(model.predict_proba(to_frame(row))[0, 1])
    return {
        "modele": req.modele,
        "probabilite_decrochage": round(proba, 4),
        "seuil": req.seuil,
        "decision": "A risque" if proba >= req.seuil else "Pas a risque",
    }


@app.get("/students/{index}", summary="Récupérer un étudiant du jeu de test")
def student(index: int):
    if not 0 <= index < len(test_df):
        raise HTTPException(status_code=404,
                            detail=f"Index entre 0 et {len(test_df) - 1}")
    row = test_df.iloc[index]
    return {"index": index,
            "etudiant": {c: float(row[c]) for c in FEATURES},
            "decrochage_reel": int(row[meta["target"]])}


@app.post(
    "/explain/shap",
    summary="Expliquer une prédiction XGBoost avec SHAP"
)
def explain_shap(req: PredictRequest):
    # Pour cette première version, SHAP explique XGBoost
    if req.modele != "xgboost":
        raise HTTPException(
            status_code=400,
            detail="Cette explication SHAP est disponible uniquement pour XGBoost."
        )

    # Vérifier que toutes les variables sont présentes
    manquantes = [
        c for c in FEATURES
        if c not in req.etudiant
    ]

    if manquantes:
        raise HTTPException(
            status_code=422,
            detail=f"Variables manquantes : {manquantes}"
        )

    # Réutiliser le prétraitement de l'API
    row = to_frame(pd.DataFrame([req.etudiant]))

    # Calculer l'explication
    engine = get_shap_engine()
    result = engine.explain_student(row.iloc[0].to_dict())

    # Calculer la décision au seuil demandé
    probability = result["probabilite"]

    return {
        "modele": "xgboost",
        "probabilite_decrochage": probability,
        "seuil": req.seuil,
        "decision": (
            "A risque"
            if probability >= req.seuil
            else "Pas a risque"
        ),
        "base_value": result["base_value"],
        "unite_shap": result["unite_shap"],
        "contributions": result["contributions"]
    }

# Module 5 : routes complementaires SHAP/LIME/importances
from api.xai_extensions import router as xai_router
app.include_router(xai_router)
