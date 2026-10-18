"""Routes supplémentaires du module 5, sans modifier les routes historiques."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from lime.lime_tabular import LimeTabularExplainer

router = APIRouter(prefix='/explain', tags=['Explainable AI'])
ART = Path(__file__).resolve().parents[1] / 'artifacts'

@router.get('/importances')
def importances():
    file=ART/'xai_importances.csv'
    if not file.exists():
        raise HTTPException(503,'Exécuter python -m scripts.prepare_xai')
    return pd.read_csv(file).replace({np.nan:None}).to_dict('records')

@lru_cache(maxsize=1)
def get_lime():
    from api.main import MODELS, FEATURES, INT_COLS, meta
    file=ART/'lime_train.csv'
    if not file.exists():
        raise HTTPException(503,'Jeu d’entraînement absent : exécuter python -m scripts.prepare_xai ou exporter X_train depuis Kaggle')
    train=pd.read_csv(file)[FEATURES]
    cat_idx=[FEATURES.index(c) for c in INT_COLS]
    expl=LimeTabularExplainer(train.to_numpy(dtype=float),feature_names=FEATURES,
                             categorical_features=cat_idx,class_names=['reste','abandon'],
                             discretize_continuous=True,random_state=42)
    return expl

@router.get('/lime/{index}')
def explain_lime(index:int):
    from api.main import MODELS,FEATURES,INT_COLS,X_test,to_frame
    if not 0 <= index < len(X_test):
        raise HTTPException(404,'Étudiant hors du jeu de test')
    row=to_frame(X_test.iloc[[index]])
    model=MODELS['xgboost']
    def predict_fn(arr):
        d=pd.DataFrame(arr,columns=FEATURES)
        d[INT_COLS]=d[INT_COLS].round().astype(int)
        return model.predict_proba(to_frame(d))
    exp=get_lime().explain_instance(row.iloc[0].to_numpy(dtype=float),
                predict_fn,num_features=len(FEATURES),num_samples=2500)
    weights={FEATURES[j]:float(value) for j,value in exp.as_map()[1]}
    return {'index':index, 'probabilite_decrochage':float(model.predict_proba(row)[0,1]),
            'contributions':weights,'conditions':[
                {'condition':str(txt),'poids':float(v)} for txt,v in exp.as_list()[:12]],
            'unite_lime':'poids locaux en probabilite',
            'note':'LIME est une approximation locale et peut varier selon les perturbations.'}
