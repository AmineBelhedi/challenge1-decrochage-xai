from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
meta = json.loads((ART / 'meta.json').read_text())
features = meta['features']
df = pd.read_csv(ART / 'test_set.csv')
X, y = df[features], df[meta['target']]
models = {name: joblib.load(ART / f'{name}_pipeline.joblib') for name in ('xgboost','logreg')}

# Mesures calculées sur les variables ORIGINALES, donc permutation du pipeline entier.
res = {}
for name, model in models.items():
    print('Permutation :', name, flush=True)
    result = permutation_importance(model, X, y, scoring='roc_auc',
                                    n_repeats=5, random_state=42, n_jobs=1)
    res['permutation_' + name] = result.importances_mean

# Les gains XGBoost des colonnes One-Hot sont additionnés par variable source.
from api.explain import ShapExplainer
eng = ShapExplainer(models['xgboost'])
res['importance_xgboost'] = models['xgboost'].named_steps['clf'].feature_importances_ @ eng.M
pd.DataFrame({'variable':features,**res}).to_csv(ART/'xai_importances.csv',index=False)
print('Importances enregistrées : artifacts/xai_importances.csv')

# Reconstituer le jeu d'entraînement selon le split EXACT du notebook.
# Si indisponible, laisser LIME désactivé au lieu d'utiliser le jeu de test.
background=ART/'lime_train.csv'
if not background.exists():
    try:
        from ucimlrepo import fetch_ucirepo
        original=fetch_ucirepo(id=697)
        full=original.data.features.copy()
        target=(original.data.targets['Target']=='Dropout').astype(int)
        X_train, X_check, _, y_check=train_test_split(
            full[features],target,test_size=.2,stratify=target,random_state=42)
        # S'assurer que le dataset reconstitué correspond aux artefacts.
        if (X_check.reset_index(drop=True).round(8).equals(X.reset_index(drop=True).round(8))
                and np.array_equal(y_check.to_numpy(),y.to_numpy())):
            X_train.to_csv(background,index=False)
            print('Données d’entraînement LIME enregistrées :',background)
        else:
            print('ATTENTION: le split UCI ne correspond pas au test existant; LIME non activé.')
    except Exception as exc:
        print('Téléchargement du train impossible:', repr(exc))
        print('Alternative Kaggle : X_train.to_csv("lime_train.csv", index=False) puis copier dans artifacts/')
else:
    print('Données LIME déjà disponibles.')
