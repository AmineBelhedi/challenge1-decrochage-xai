# Prédire le décrochage étudiant : classification explicable

Projet Challenge 1, Data Mining (II3, ENSI). Cas métier 10 : *Predict Students' Dropout and Academic Success* (UCI, identifiant 697).

Auteur : Mohamed Amine Belhedi

## Contenu du projet

```
challenge1-decrochage-xai/
├── api/main.py              API FastAPI (prédiction, métriques)
├── app/streamlit_app.py     Application Streamlit (modules 1 à 4, appel de l'API)
├── artifacts/               5 modèles entraînés (.joblib), jeu de test, meta.json
├── notebook/                Notebook Kaggle (exploration, préparation, modèles, évaluation)
├── requirements.txt         Dépendances (versions identiques à celles du notebook)
└── README.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 ou 3.13. Les versions de scikit-learn (1.6.1), XGBoost (3.4.1) et pandas (2.3.3) sont fixées : les modèles sauvegardés doivent être relus avec les mêmes versions que celles utilisées pour les entraîner.

## Lancer l'API

```bash
uvicorn api.main:app --reload
```

Documentation interactive : http://127.0.0.1:8000/docs

| Route | Rôle |
|---|---|
| `GET /health` | Vérifier que l'API fonctionne |
| `GET /models` | Lister les 5 modèles |
| `GET /metrics` | Métriques d'un modèle sur le jeu de test, pour un seuil donné |
| `POST /predict` | Probabilité de décrochage et décision pour un étudiant |
| `GET /students/{index}` | Un étudiant du jeu de test |

## Lancer l'application

Dans un second terminal, l'API étant lancée :

```bash
streamlit run app/streamlit_app.py
```

Parcours : *1. Données* (téléchargement automatique UCI), *2. Préparation*, *3. Modélisation*, *4. Évaluation*, *6. API et prédiction*. L'onglet *5. Explicabilité* (SHAP et LIME) sera ajouté ensuite.

## Résultats (jeu de test : 885 étudiants, 32,1 % de décrocheurs)

| Modèle | AUC test |
|---|---|
| XGBoost | 0,936 |
| Régression logistique | 0,932 |
| Random Forest | 0,926 |
| k-NN | 0,887 |
| Arbre de décision | 0,885 |

XGBoost et la régression logistique sont équivalents : l'écart (0,004) est inférieur à l'incertitude d'un jeu de test de cette taille. Seuil de décision recommandé : **0,3**. Avec XGBoost, il détecte 85,6 % des décrocheurs (précision 78,6 %), contre 75,7 % au seuil de 0,5 (précision 88,8 %), parce qu'un décrocheur non détecté coûte plus cher qu'une fausse alerte.

## Limites

- Les variables du 2e semestre sont les plus prédictives mais arrivent tard : le modèle détecte bien le décrochage, pas forcément assez tôt pour une alerte précoce.
- Les étudiants encore inscrits (*Enrolled*) sont classés parmi les non-décrocheurs, alors que leur issue finale est inconnue.


## Liens

- Dépôt GitHub : https://github.com/AmineBelhedi/challenge1-decrochage-xai

