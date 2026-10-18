"""Application web de classification explicable : décrochage étudiant.

Challenge 1, Data Mining, ENSI.
Modules 1 à 4 (données, préparation, modélisation, évaluation) et connexion à l'API
FastAPI (module 6). Le module 5 (SHAP et LIME) sera ajouté ensuite.

Lancer :  streamlit run app/streamlit_app.py
"""
import os
import time
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from pathlib import Path
import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve,
                             precision_recall_curve, confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="Décrochage étudiant : classification explicable",
                   layout="wide")

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
UCI_ID = 697  # Predict Students' Dropout and Academic Success
# Colonnes codées par des entiers mais qui sont des catégories (voir le notebook)
UCI_CATEGORICAL = ["Marital Status", "Application mode", "Course", "Previous qualification",
                   "Nacionality", "Mother's qualification", "Father's qualification",
                   "Mother's occupation", "Father's occupation"]
MODEL_NAMES = ["Régression logistique", "Arbre de décision", "k-NN", "Random Forest", "XGBoost"]
MANUAL = "Réglages manuels (curseurs)"
AUTO = "Optimisation automatique (GridSearchCV)"
DROP_ROWS = "Supprimer les lignes incomplètes"
IMPUTE = "Imputer les valeurs manquantes"
SCALERS = {"Standardisation (StandardScaler)": StandardScaler,
           "Min-Max (MinMaxScaler)": MinMaxScaler,
           "Aucune": None}

DEFAULT_HP = {
    "Régression logistique": {"C": 1.0},
    "Arbre de décision": {"max_depth": 5, "min_samples_leaf": 20},
    "k-NN": {"n_neighbors": 21},
    "Random Forest": {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 3},
    "XGBoost": {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
}
# Petites grilles pour l'optimisation automatique (validation croisée, critère AUC)
GRIDS = {
    "Régression logistique": {"C": [0.01, 0.1, 1, 10]},
    "Arbre de décision": {"max_depth": [3, 5, 8], "min_samples_leaf": [5, 20]},
    "k-NN": {"n_neighbors": [5, 11, 21]},
    "Random Forest": {"max_depth": [8, 16], "min_samples_leaf": [1, 5]},
    "XGBoost": {"n_estimators": [200, 400], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
}


# ----------------------------------------------------------------------------
# Fonctions utilitaires
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_uci() -> pd.DataFrame:
    """Télécharge le jeu UCI automatiquement (aucune manipulation manuelle)."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=UCI_ID)
    df = ds.data.features.copy()
    df["Target"] = ds.data.targets["Target"].values
    return df


def set_dataset(df: pd.DataFrame, source: str) -> None:
    """Enregistre un nouveau jeu de données et efface tout ce qui en dépend."""
    st.session_state["df"] = df
    st.session_state["source"] = source
    st.session_state["ds_version"] = st.session_state.get("ds_version", 0) + 1
    for key in ("prep", "results"):
        st.session_state.pop(key, None)


def binarize(df: pd.DataFrame, target: str, positive):
    """Cible binaire : 1 si la valeur est la classe d'intérêt, 0 sinon."""
    d = df[df[target].notna()]
    y = (d[target] == positive).astype(int)
    return d, y


def class_colors(positive) -> dict:
    """Orange pour la classe positive, bleu pour les autres (lisible sur fond sombre)."""
    return {f"{positive} (positive)": "#e4572e", "Autres": "#2e86ab"}


def is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def build_preprocessor(num_cols, cat_cols, bin_cols, num_strategy, scaler_name):
    scaler_cls = SCALERS[scaler_name]
    scaler = scaler_cls() if scaler_cls else "passthrough"
    parts = []
    if num_cols:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy=num_strategy)),
                                       ("sc", scaler)]), num_cols))
    if cat_cols:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                             sparse_output=False))]), cat_cols))
    if bin_cols:
        parts.append(("bin", SimpleImputer(strategy="most_frequent"), bin_cols))
    return ColumnTransformer(parts)


def make_estimator(name, hp, seed, n_jobs):
    p = {**DEFAULT_HP[name], **hp}
    if name == "Régression logistique":
        return LogisticRegression(C=p["C"], max_iter=3000)
    if name == "Arbre de décision":
        return DecisionTreeClassifier(max_depth=p["max_depth"],
                                      min_samples_leaf=p["min_samples_leaf"], random_state=seed)
    if name == "k-NN":
        return KNeighborsClassifier(n_neighbors=p["n_neighbors"], n_jobs=n_jobs)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      min_samples_leaf=p["min_samples_leaf"],
                                      random_state=seed, n_jobs=n_jobs)
    return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                         learning_rate=p["learning_rate"], subsample=p["subsample"],
                         eval_metric="logloss", random_state=seed, n_jobs=n_jobs)


def metrics_at(y, proba, thr):
    """Toutes les métriques pour un seuil de décision donné."""
    y = np.asarray(y)
    pred = (proba >= thr).astype(int)
    auc = roc_auc_score(y, proba)
    fpr, tpr, _ = roc_curve(y, proba)
    return {
        "AUC-ROC": auc,
        "PR-AUC": average_precision_score(y, proba),
        "Gini": 2 * auc - 1,
        "KS": float(np.max(tpr - fpr)),
        "Accuracy": accuracy_score(y, pred),
        "Précision": precision_score(y, pred, zero_division=0),
        "Rappel": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "Alertes": int(pred.sum()),
    }


def curve_vs_threshold(y, proba):
    """Précision, rappel et F1 pour 99 seuils (calcul vectorisé)."""
    y = np.asarray(y)
    thr = np.linspace(0.01, 0.99, 99)
    pred = proba[None, :] >= thr[:, None]
    tp = (pred & (y == 1)).sum(axis=1)
    fp = (pred & (y == 0)).sum(axis=1)
    fn = (y == 1).sum() - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return thr, precision, recall, f1


def get_prep():
    """Renvoie la préparation courante si elle correspond encore à la cible choisie."""
    prep = st.session_state.get("prep")
    if prep is None:
        return None
    if prep["sig"] != (st.session_state.get("target"), st.session_state.get("positive")):
        return None
    return prep


# ----------------------------------------------------------------------------
# Barre latérale
# ----------------------------------------------------------------------------
def sidebar_progress(container):
    """Avancement : appelé après les onglets pour refléter l'état à jour."""
    steps = [("Données chargées", "df" in st.session_state),
             ("Préparation appliquée", get_prep() is not None),
             ("Modèles entraînés", "results" in st.session_state)]
    with container:
        st.header("Avancement")
        for label, done in steps:
            st.markdown(("✓ " if done else "– ") + label)


def sidebar_api():
    st.sidebar.divider()
    st.sidebar.header("API FastAPI")
    st.sidebar.text_input("Adresse de l'API",
                          value=os.environ.get("API_URL", "http://127.0.0.1:8000"),
                          key="api_url", help="L'API doit être lancée : uvicorn api.main:app")


# ----------------------------------------------------------------------------
# Module 1 : Données
# ----------------------------------------------------------------------------
def rate_by_variable(d, y, col, bins=10):
    s = d[col]
    if is_num(s) and s.nunique() > 12:
        group = pd.qcut(s, q=min(bins, s.nunique()), duplicates="drop")
    else:
        group = s
    t = (pd.DataFrame({"groupe": group, "y": y.values})
         .groupby("groupe", observed=True)["y"].agg(["mean", "count"]).reset_index())
    t["groupe"] = t["groupe"].astype(str)
    return t


def tab_data():
    st.subheader("Module 1 : données")
    source = st.radio("Source des données",
                      ["Jeu UCI : décrochage et réussite des étudiants", "Importer un fichier CSV"],
                      horizontal=True, key="source_choice")

    if source.startswith("Jeu UCI"):
        st.caption("Téléchargement automatique depuis l'UCI avec ucimlrepo "
                   "(4 424 étudiants, 36 variables).")
        if st.button("Télécharger le jeu de données", key="btn_load_uci", type="primary"):
            with st.spinner("Téléchargement en cours..."):
                try:
                    set_dataset(load_uci(), "uci")
                except Exception as exc:  # réseau, serveur UCI lent...
                    st.error(f"Le téléchargement a échoué ({exc}). Réessayez, "
                             "ou importez le fichier CSV.")
    else:
        file = st.file_uploader("Fichier CSV", type=["csv"], key="csv_file")
        sep = st.selectbox("Séparateur", [",", ";", "\t"], key="csv_sep",
                           format_func=lambda s: {",": "virgule", ";": "point-virgule",
                                                  "\t": "tabulation"}[s])
        if file is not None:
            sig = (file.name, file.size, sep)
            if st.session_state.get("csv_sig") != sig:
                try:
                    file.seek(0)
                    set_dataset(pd.read_csv(file, sep=sep), "csv")
                    st.session_state["csv_sig"] = sig
                except Exception as exc:
                    st.error(f"Lecture impossible : {exc}")
        st.caption("Indiquez ensuite la variable cible ci-dessous.")

    df = st.session_state.get("df")
    if df is None:
        st.info("Chargez un jeu de données pour commencer.")
        return

    ver = st.session_state["ds_version"]
    cols = list(df.columns)

    # --- Variable cible ---
    st.markdown("#### Variable cible")
    default_target = "Target" if st.session_state["source"] == "uci" and "Target" in cols else cols[-1]
    c1, c2 = st.columns(2)
    target = c1.selectbox("Variable à prédire", cols, index=cols.index(default_target),
                          key=f"target_{ver}")
    values = sorted(df[target].dropna().unique().tolist(), key=str)
    if not 2 <= len(values) <= 20:
        st.error("La cible doit avoir entre 2 et 20 valeurs distinctes (classification).")
        return
    if st.session_state["source"] == "uci" and target == "Target" and "Dropout" in values:
        default_pos = "Dropout"
    else:
        default_pos = 1 if 1 in values else values[-1]
    positive = c2.selectbox("Classe d'intérêt (positive)", values, index=values.index(default_pos),
                            key=f"positive_{ver}")
    st.session_state["target"], st.session_state["positive"] = target, positive
    if len(values) > 2:
        st.caption(f"La cible a {len(values)} classes : elles sont ramenées à deux, "
                   f"« {positive} » contre toutes les autres.")

    d, y = binarize(df, target, positive)
    feat = [c for c in df.columns if c != target]
    n, npos = len(d), int(y.sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observations", f"{n:,}".replace(",", " "))
    m2.metric("Variables explicatives", len(feat))
    m3.metric(f"Classe positive ({positive})", f"{npos:,}".replace(",", " "))
    m4.metric("Part de la classe positive", f"{npos / n:.1%}")

    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Répartition des classes**")
        share = pd.DataFrame({"Classe": [f"{positive} (positive)", "Autres"],
                              "Effectif": [npos, n - npos]})
        fig = px.bar(share, x="Classe", y="Effectif", text="Effectif", color="Classe",
                     color_discrete_map=class_colors(positive))
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**Valeurs manquantes**")
        miss = df[feat].isna().sum()
        miss = miss[miss > 0]
        if miss.empty:
            st.success("Aucune valeur manquante dans les variables explicatives.")
        else:
            st.dataframe(pd.DataFrame({"Manquantes": miss, "Part": miss / len(df)})
                         .sort_values("Manquantes", ascending=False),
                         column_config={"Part": st.column_config.NumberColumn(format="percent")},
                         width="stretch")

    with st.expander("Aperçu des données", expanded=True):
        st.dataframe(df.head(50), width="stretch")
    with st.expander("Statistiques descriptives"):
        st.dataframe(df[[c for c in feat if is_num(df[c])]].describe().T, width="stretch")
        non_num = [c for c in feat if not is_num(df[c])]
        if non_num:
            st.dataframe(df[non_num].describe().T, width="stretch")

    # --- Visualisations exploratoires ---
    st.markdown("#### Exploration")
    t_dist, t_corr, t_rate = st.tabs(["Distribution", "Corrélations",
                                      "Taux de la classe positive"])
    with t_dist:
        col = st.selectbox("Variable", feat, key=f"dist_{ver}")
        tmp = d[[col]].copy()
        tmp["Classe"] = np.where(y == 1, f"{positive} (positive)", "Autres")
        fig = px.histogram(tmp, x=col, color="Classe", barmode="overlay", opacity=0.65,
                           nbins=40, histnorm="percent",
                           color_discrete_map=class_colors(positive))
        fig.update_layout(yaxis_title="Part de la classe (%)", height=380)
        st.plotly_chart(fig, width="stretch")
    with t_corr:
        num = [c for c in feat if is_num(d[c])]
        corr_t = d[num].corrwith(y).dropna().sort_values(key=lambda s: s.abs(), ascending=False)
        top = corr_t.head(15)[::-1]
        fig = px.bar(x=top.values, y=top.index, orientation="h", color=top.values,
                     color_continuous_scale="RdBu_r", range_color=(-1, 1),
                     labels={"x": "Corrélation avec la cible", "y": "", "color": ""})
        fig.update_layout(height=450, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.caption("Corrélation linéaire avec la cible. Pour une catégorie codée par un nombre "
                   "(un code de filière, par exemple), cette valeur n'a pas de sens.")
        sel = st.multiselect("Variables de la matrice de corrélation", num,
                             default=list(corr_t.head(8).index), key=f"corr_{ver}_{target}")
        if len(sel) >= 2:
            fig = px.imshow(d[sel].corr(), text_auto=".2f", zmin=-1, zmax=1,
                            color_continuous_scale="RdBu_r", aspect="auto")
            fig.update_layout(height=520)
            st.plotly_chart(fig, width="stretch")
    with t_rate:
        col = st.selectbox("Variable", feat, key=f"rate_{ver}")
        t = rate_by_variable(d, y, col)
        fig = px.bar(t, x="groupe", y="mean", text="count",
                     labels={"groupe": col, "mean": f"Taux de « {positive} »"})
        fig.add_hline(y=npos / n, line_dash="dash", line_color="gray",
                      annotation_text="Taux global")
        fig.update_traces(texttemplate="n=%{text}", textposition="outside")
        fig.update_layout(height=400, yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.caption("Chaque barre est le taux de la classe positive dans le groupe ; "
                   "n est l'effectif du groupe.")


# ----------------------------------------------------------------------------
# Module 2 : Préparation
# ----------------------------------------------------------------------------
def tab_prep():
    st.subheader("Module 2 : préparation")
    df = st.session_state.get("df")
    if df is None:
        st.info("Chargez d'abord les données (onglet « 1. Données »).")
        return
    ver = st.session_state["ds_version"]
    target, positive = st.session_state["target"], st.session_state["positive"]
    st.markdown(f"Cible : **{target}**, classe positive : **{positive}** "
                "(modifiable dans l'onglet « 1. Données »).")

    all_feat = [c for c in df.columns if c != target]
    feats = st.multiselect("Variables explicatives", all_feat, default=all_feat,
                           key=f"feats_{ver}_{target}")
    st.caption("Retirez toute variable qui n'est connue qu'après l'événement à prédire : "
               "c'est une fuite de données.")
    if not feats:
        st.warning("Sélectionnez au moins une variable.")
        return

    sig = abs(hash((target, tuple(feats))))
    if st.session_state["source"] == "uci":
        cat_default = [c for c in UCI_CATEGORICAL if c in feats]
    else:
        cat_default = [c for c in feats if not is_num(df[c])]
    cat_cols = st.multiselect("Variables catégorielles (encodage One-Hot)", feats,
                              default=cat_default, key=f"cats_{ver}_{sig}")
    forced = [c for c in feats if not is_num(df[c]) and c not in cat_cols]
    if forced:
        cat_cols += forced
        st.caption(f"Variables de type texte ajoutées aux catégorielles : {', '.join(forced)}.")
    rest = [c for c in feats if c not in cat_cols]
    bin_cols = [c for c in rest if df[c].nunique(dropna=True) == 2]
    num_cols = [c for c in rest if c not in bin_cols]

    a, b, c = st.columns(3)
    a.metric("Numériques (normalisées)", len(num_cols))
    b.metric("Catégorielles (One-Hot)", len(cat_cols))
    c.metric("Binaires (inchangées)", len(bin_cols))

    st.markdown("#### Traitements")
    c1, c2, c3 = st.columns(3)
    missing_mode = c1.radio("Valeurs manquantes", [IMPUTE, DROP_ROWS], key="missing_mode")
    num_strategy = c1.selectbox("Imputation des variables numériques", ["median", "mean"],
                                format_func=lambda s: {"median": "Médiane", "mean": "Moyenne"}[s],
                                key="num_strategy", disabled=missing_mode == DROP_ROWS)
    c1.caption("Les catégories manquantes sont remplacées par la valeur la plus fréquente.")
    scaler_name = c2.selectbox("Normalisation des variables numériques", list(SCALERS),
                               key="scaler_name")
    c2.caption("Utile pour la régression logistique et k-NN ; sans effet sur les arbres.")
    test_size = c3.slider("Part du jeu de test (%)", 10, 40, 20, 5, key="test_size") / 100
    seed = c3.number_input("Graine aléatoire", 0, 9999, 42, key="seed")

    if st.button("Appliquer la préparation", key="btn_apply_prep", type="primary"):
        d, y = binarize(df, target, positive)
        X = d[feats]
        if missing_mode == DROP_ROWS:
            keep = X.notna().all(axis=1)
            X, y = X[keep], y[keep]
        if y.nunique() < 2 or y.value_counts().min() < 10:
            st.error("Il reste trop peu d'observations dans une des deux classes.")
            return
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=int(seed))
        pre = build_preprocessor(num_cols, cat_cols, bin_cols, num_strategy, scaler_name)
        n_out = clone(pre).fit(X_train).transform(X_train).shape[1]
        st.session_state["prep"] = dict(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            preprocessor=pre, n_in=len(feats), n_out=n_out, seed=int(seed),
            sig=(target, positive))
        st.session_state.pop("results", None)

    prep = get_prep()
    if st.session_state.get("prep") is not None and prep is None:
        st.warning("La cible a changé : appliquez de nouveau la préparation.")
    if prep:
        st.success("Préparation appliquée. Les transformations sont apprises sur le jeu "
                   "d'entraînement uniquement, puis appliquées au test (pas de fuite de données).")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Entraînement", f"{len(prep['X_train']):,}".replace(",", " "))
        k2.metric("Test", f"{len(prep['X_test']):,}".replace(",", " "))
        k3.metric("Part positive (train / test)",
                  f"{prep['y_train'].mean():.1%} / {prep['y_test'].mean():.1%}")
        k4.metric("Colonnes après encodage", f"{prep['n_in']} → {prep['n_out']}")


# ----------------------------------------------------------------------------
# Module 3 : Modélisation
# ----------------------------------------------------------------------------
def hp_widgets(name):
    def k(p):
        return f"hp_{name}_{p}"
    if name == "Régression logistique":
        return {"C": st.select_slider("C (inverse de la régularisation)",
                                      [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0, key=k("C"))}
    if name == "Arbre de décision":
        return {"max_depth": st.slider("Profondeur maximale", 1, 20, 5, key=k("md")),
                "min_samples_leaf": st.slider("Taille minimale d'une feuille", 1, 50, 20,
                                              key=k("msl"))}
    if name == "k-NN":
        return {"n_neighbors": st.slider("Nombre de voisins (k)", 1, 51, 21, key=k("k"))}
    if name == "Random Forest":
        return {"n_estimators": st.slider("Nombre d'arbres", 50, 500, 200, 50, key=k("n")),
                "max_depth": st.slider("Profondeur maximale", 2, 30, 12, key=k("md")),
                "min_samples_leaf": st.slider("Taille minimale d'une feuille", 1, 20, 3,
                                              key=k("msl"))}
    return {"n_estimators": st.slider("Nombre d'arbres", 50, 1000, 400, 50, key=k("n")),
            "max_depth": st.slider("Profondeur maximale", 1, 10, 3, key=k("md")),
            "learning_rate": st.select_slider("Taux d'apprentissage",
                                              [0.01, 0.02, 0.05, 0.1, 0.2, 0.3], value=0.05,
                                              key=k("lr")),
            "subsample": st.slider("Part des étudiants vus par chaque arbre", 0.5, 1.0, 0.8, 0.05,
                                   key=k("ss"))}


def clean_params(params):
    return {k.replace("clf__", ""): (round(float(v), 4) if isinstance(v, (float, np.floating))
                                     else (int(v) if isinstance(v, (int, np.integer)) else v))
            for k, v in params.items()}


def train_all(prep, selected, hp, mode, cv_folds):
    seed = prep["seed"]
    results = {}
    bar = st.progress(0.0, text="Entraînement...")
    for i, name in enumerate(selected):
        bar.progress(i / len(selected), text=f"Entraînement : {name}")
        auto = mode == AUTO
        est = make_estimator(name, hp.get(name, {}), seed, n_jobs=1 if auto else -1)
        pipe = Pipeline([("prep", clone(prep["preprocessor"])), ("clf", est)])
        t0 = time.time()
        cv_auc, params = None, clean_params(hp.get(name, DEFAULT_HP[name]))
        if auto:
            grid = {f"clf__{k}": v for k, v in GRIDS[name].items()}
            search = GridSearchCV(pipe, grid, scoring="roc_auc", n_jobs=-1,
                                  cv=StratifiedKFold(cv_folds, shuffle=True, random_state=seed))
            search.fit(prep["X_train"], prep["y_train"])
            fitted, cv_auc = search.best_estimator_, float(search.best_score_)
            params = {**clean_params({k: v for k, v in DEFAULT_HP[name].items()
                                      if k not in GRIDS[name]}),
                      **clean_params(search.best_params_)}
        else:
            fitted = pipe.fit(prep["X_train"], prep["y_train"])
        p_test = fitted.predict_proba(prep["X_test"])[:, 1]
        p_train = fitted.predict_proba(prep["X_train"])[:, 1]
        results[name] = dict(pipe=fitted, p_test=p_test, cv_auc=cv_auc, params=params,
                             auc_train=float(roc_auc_score(prep["y_train"], p_train)),
                             auc_test=float(roc_auc_score(prep["y_test"], p_test)),
                             seconds=time.time() - t0)
    bar.progress(1.0, text="Entraînement terminé")
    return results


def tab_models():
    st.subheader("Module 3 : modélisation et comparaison")
    prep = get_prep()
    if prep is None:
        st.info("Appliquez d'abord la préparation (onglet « 2. Préparation »).")
        return

    selected = st.multiselect("Modèles à entraîner", MODEL_NAMES, default=MODEL_NAMES,
                              key="sel_models")
    mode = st.radio("Hyperparamètres", [MANUAL, AUTO], horizontal=True, key="hp_mode")
    hp, cv_folds = {}, 3
    if mode == MANUAL:
        cols = st.columns(2)
        for i, name in enumerate(selected):
            with cols[i % 2].expander(f"Hyperparamètres : {name}"):
                hp[name] = hp_widgets(name)
    else:
        cv_folds = st.select_slider("Plis de la validation croisée", [3, 5], value=3,
                                    key="cv_folds")
        st.caption("Chaque modèle essaie toutes les combinaisons de sa grille en validation "
                   "croisée sur le jeu d'entraînement ; la meilleure est conservée "
                   "(critère : AUC-ROC).")
        st.dataframe(pd.DataFrame({"Modèle": list(GRIDS),
                                   "Grille": [str(g) for g in GRIDS.values()]}),
                     hide_index=True, width="stretch")

    if st.button("Entraîner les modèles", key="btn_train", type="primary", disabled=not selected):
        st.session_state["results"] = train_all(prep, selected, hp, mode, cv_folds)

    results = st.session_state.get("results")
    if not results:
        return
    st.markdown("#### Tableau comparatif")
    table = pd.DataFrame([{
        "Modèle": n,
        "AUC entraînement": r["auc_train"],
        "AUC validation croisée": r["cv_auc"],
        "AUC test": r["auc_test"],
        "Écart entraînement - test": r["auc_train"] - r["auc_test"],
        "Temps (s)": r["seconds"],
        "Hyperparamètres": str(r["params"]),
    } for n, r in results.items()]).sort_values("AUC test", ascending=False)
    if table["AUC validation croisée"].isna().all():
        table = table.drop(columns=["AUC validation croisée"])  # vide en mode manuel
    fmt = st.column_config.NumberColumn(format="%.3f")
    cfg = {c: fmt for c in ["AUC entraînement", "AUC validation croisée", "AUC test",
                            "Écart entraînement - test"] if c in table.columns}
    cfg["Temps (s)"] = st.column_config.NumberColumn(format="%.1f")
    st.dataframe(table, hide_index=True, width="stretch", column_config=cfg)
    st.caption("Un grand écart entre l'AUC d'entraînement et l'AUC de test signale un "
               "surapprentissage : le modèle a mémorisé les données d'entraînement. "
               "Les métriques dépendantes du seuil sont dans l'onglet « 4. Évaluation ».")


# ----------------------------------------------------------------------------
# Module 4 : Évaluation
# ----------------------------------------------------------------------------
def tab_eval():
    st.subheader("Module 4 : évaluation")
    prep, results = get_prep(), st.session_state.get("results")
    if prep is None or not results:
        st.info("Entraînez d'abord des modèles (onglet « 3. Modélisation »).")
        return
    y_test = prep["y_test"].values
    positive = st.session_state["positive"]

    c1, c2 = st.columns([2, 1])
    thr = c1.slider("Seuil de décision : alerte si probabilité ≥ seuil", 0.01, 0.99, 0.50, 0.01,
                    key="threshold")
    chosen = c2.selectbox("Modèle détaillé", list(results), key="eval_model",
                          index=list(results).index("XGBoost") if "XGBoost" in results else 0)

    # Effet du seuil en direct, comparé au seuil 0,5
    now = metrics_at(y_test, results[chosen]["p_test"], thr)
    ref = metrics_at(y_test, results[chosen]["p_test"], 0.5)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rappel", f"{now['Rappel']:.1%}", f"{now['Rappel'] - ref['Rappel']:+.1%} vs seuil 0,5")
    k2.metric("Précision", f"{now['Précision']:.1%}",
              f"{now['Précision'] - ref['Précision']:+.1%} vs seuil 0,5")
    k3.metric("F1", f"{now['F1']:.3f}", f"{now['F1'] - ref['F1']:+.3f} vs seuil 0,5")
    k4.metric("Étudiants signalés", now["Alertes"], f"{now['Alertes'] - ref['Alertes']:+d} vs seuil 0,5",
              delta_color="off")

    st.markdown(f"#### Métriques de tous les modèles (seuil {thr:.2f}, classe d'intérêt : {positive})")
    rows = [{"Modèle": n, **metrics_at(y_test, r["p_test"], thr)} for n, r in results.items()]
    table = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False)
    fmt = st.column_config.NumberColumn(format="%.3f")
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: fmt for c in ["AUC-ROC", "PR-AUC", "Gini", "KS", "Accuracy",
                                                 "Précision", "Rappel", "F1"]})
    st.caption("Gini = 2 × AUC − 1. KS = écart maximal entre les taux de vrais et de faux "
               "positifs. Le PR-AUC est la métrique de référence quand les classes sont "
               "déséquilibrées : un modèle sans compétence obtient la part de la classe positive.")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for n, r in results.items():
            fpr, tpr, _ = roc_curve(y_test, r["p_test"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{n} (AUC {r['auc_test']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard",
                                 line=dict(dash="dash", color="gray")))
        fig.update_layout(title="Courbes ROC superposées", height=430,
                          xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs",
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure()
        for n, r in results.items():
            pr, rc, _ = precision_recall_curve(y_test, r["p_test"])
            ap = average_precision_score(y_test, r["p_test"])
            fig.add_trace(go.Scatter(x=rc, y=pr, mode="lines", name=f"{n} (PR-AUC {ap:.3f})"))
        fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="gray",
                      annotation_text="Hasard")
        fig.update_layout(title="Courbes précision-rappel", height=430, xaxis_title="Rappel",
                          yaxis_title="Précision", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        pred = (results[chosen]["p_test"] >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        fig = px.imshow([[tn, fp], [fn, tp]], text_auto=True, color_continuous_scale="Blues",
                        x=["Prédit : autres", f"Prédit : {positive}"],
                        y=["Réel : autres", f"Réel : {positive}"])
        fig.update_layout(title=f"Matrice de confusion : {chosen}", height=380,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{fn} cas positifs non détectés, {fp} fausses alertes.")
    with right:
        t_, p_, r_, f_ = curve_vs_threshold(y_test, results[chosen]["p_test"])
        fig = go.Figure()
        for label, vals in [("Précision", p_), ("Rappel", r_), ("F1", f_)]:
            fig.add_trace(go.Scatter(x=t_, y=vals, mode="lines", name=label))
        fig.add_vline(x=thr, line_dash="dash", line_color="gray")
        fig.update_layout(title=f"Effet du seuil : {chosen}", height=380, xaxis_title="Seuil",
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
        st.caption("Baisser le seuil augmente le rappel (moins de cas ratés) mais diminue la "
                   "précision (plus de fausses alertes). Le bon seuil dépend du coût de chaque erreur.")


# ----------------------------------------------------------------------------
# Module 5 : Explicabilité (à venir)
# ----------------------------------------------------------------------------

@st.cache_data
def load_global_shap():
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts"

    # Contributions SHAP précalculées
    with np.load(
        artifacts / "shap_global.npz",
        allow_pickle=False
    ) as data:
        values = data["values"].copy()
        feature_names = data["feature_names"].tolist()
        base_value = float(data["base_value"])

    # Importances globales
    importance = pd.read_csv(
        artifacts / "shap_global_importance.csv"
    )

    # Valeurs originales des étudiants
    df = pd.read_csv(artifacts / "test_set.csv")
    X = df[feature_names].copy()

    return values, feature_names, base_value, importance, X

def tab_xai():
    st.subheader("Module 5 : Explainable AI")
    st.caption(
        "Comprendre les facteurs qui influencent "
        "les prédictions de décrochage étudiant."
    )


    # ===== EXPLICATIONS GLOBALES =====
    st.markdown("### Explications globales du modèle")

    try:
        (
            global_values,
            feature_names,
            base_global,
            global_importance,
            X_global
        ) = load_global_shap()

    except (OSError, ValueError, KeyError) as e:
        st.error(f"Impossible de charger les données SHAP : {e}")
        return

    # 1. Importance globale
    st.markdown("#### Importance des variables")

    top_imp = global_importance.head(12).iloc[::-1]

    fig_global = px.bar(
        top_imp,
        x="importance_shap",
        y="variable",
        orientation="h",
        title="Variables les plus influentes selon SHAP",
        labels={
            "importance_shap": "Importance moyenne |SHAP|",
            "variable": ""
        }
    )

    fig_global.update_layout(height=550)

    st.plotly_chart(fig_global, width="stretch")

    # 2. Beeswarm
    st.markdown("#### SHAP Beeswarm")

    st.caption(
        "Chaque point représente un étudiant. "
        "La position horizontale indique l'effet "
        "sur la prédiction, exprimé en log-odds."
    )

    # Une catégorie codée par un nombre n'a pas
    # nécessairement d'ordre quantitatif.
    categorical_cols = [
        "Marital Status",
        "Application mode",
        "Course",
        "Previous qualification",
        "Nacionality",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation"
    ]

    X_color = X_global.copy()

    for col in categorical_cols:
        if col in X_color.columns:
            X_color[col] = np.nan

    # Création de l'objet Explanation
    global_explanation = shap.Explanation(
        values=global_values,
        base_values=np.full(
            len(global_values),
            base_global
        ),
        data=X_color.to_numpy(dtype=float),
        feature_names=feature_names
    )

    plt.figure(figsize=(11, 7))

    shap.plots.beeswarm(
        global_explanation,
        max_display=15,
        show=False
    )

    fig_beeswarm = plt.gcf()

    st.pyplot(fig_beeswarm)
    plt.close(fig_beeswarm)

    # 3. Graphiques de dépendance
    st.markdown("#### Dépendance SHAP")

    st.caption(
        "Observer comment la contribution d'une "
        "variable évolue selon sa valeur."
    )

    numeric_features = [
        c for c in global_importance["variable"]
        if c not in categorical_cols
    ]

    selected_feature = st.selectbox(
        "Variable à analyser",
        numeric_features,
        key="xai_dependence_feature"
    )

    fig_dep, ax = plt.subplots(figsize=(9, 5))

    idx_feature = feature_names.index(selected_feature)

    ax.scatter(
        X_global[selected_feature],
        global_values[:, idx_feature],
        alpha=0.45,
        s=18
    )

    ax.axhline(0, linestyle="--", color="gray")

    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Contribution SHAP (log-odds)")
    ax.set_title(
        f"Effet de {selected_feature} sur le risque prédit"
    )

    st.pyplot(fig_dep)
    plt.close(fig_dep)

    st.divider()
    # ===== FIN EXPLICATIONS GLOBALES =====

    from xai_extensions_ui import render_extra_xai
    render_extra_xai(api_call)

    st.markdown("### Explication individuelle avec SHAP")

    # Choisir un étudiant dans le jeu de test
    idx = int(st.number_input(
        "Numéro de l'étudiant",
        min_value=0,
        max_value=884,
        value=0,
        step=1,
        key="xai_student_idx"
    ))

    seuil = st.slider(
        "Seuil de décision",
        min_value=0.01,
        max_value=0.99,
        value=0.30,
        step=0.01,
        key="xai_threshold"
    )

    if st.button("Expliquer cet étudiant", key="xai_explain_btn"):

        # 1. Récupérer les caractéristiques
        student, err = api_call(
            "GET", f"/students/{idx}"
        )

        if err:
            st.error(err)
            return

        # 2. Demander l'explication SHAP à FastAPI
        payload = {
            "modele": "xgboost",
            "seuil": seuil,
            "etudiant": student["etudiant"]
        }

        with st.spinner("Calcul des contributions SHAP..."):
            result, err = api_call(
                "POST",
                "/explain/shap",
                json=payload
            )

        if err:
            st.error(err)
            return

        # 3. Afficher le risque et la décision
        p = result["probabilite_decrochage"]

        c1, c2 = st.columns(2)
        c1.metric("Probabilité de décrochage", f"{p:.2%}")
        c2.metric("Seuil de décision", f"{seuil:.0%}")

        if p >= seuil:
            st.error("Décision : étudiant à risque")
        else:
            st.success("Décision : étudiant sans alerte")

        # 4. Récupérer les contributions SHAP
        contributions = pd.Series(
            result["contributions"],
            dtype=float
        )

        # Les 10 facteurs ayant le plus d'impact
        top = contributions.reindex(
            contributions.abs()
            .sort_values(ascending=False)
            .head(10)
            .index
        )

        # 5. Construire le graphique
        plot_data = pd.DataFrame({
            "Variable": top.index,
            "Contribution": top.values
        })

        plot_data = plot_data.sort_values(
            "Contribution"
        )

        plot_data["Effet"] = np.where(
            plot_data["Contribution"] > 0,
            "Augmente le risque",
            "Diminue le risque"
        )

        fig = px.bar(
            plot_data,
            x="Contribution",
            y="Variable",
            orientation="h",
            color="Effet",
            color_discrete_map={
                "Augmente le risque": "#D9534F",
                "Diminue le risque": "#2E86AB"
            },
            title="Les 10 principales contributions SHAP"
        )

        fig.update_layout(
            height=520,
            xaxis_title="Contribution SHAP (log-odds)",
            yaxis_title="",
            legend_title=""
        )

        st.plotly_chart(fig, width="stretch")


        # 7. Graphique SHAP waterfall
        st.markdown("### Décomposition SHAP (Waterfall)")

        st.caption(
            "Le waterfall montre comment chaque variable "
            "déplace la prédiction par rapport à la valeur "
            "de référence du modèle."
        )

        # Récupérer les 36 contributions dans leur ordre
        shap_contributions = result["contributions"]

        names = list(shap_contributions.keys())
        values = np.array(
            list(shap_contributions.values()),
            dtype=float
        )

        # Construire un objet SHAP Explanation
        explanation = shap.Explanation(
            values=values,
            base_values=float(result["base_value"]),
            feature_names=names
        )

        # Générer le véritable waterfall SHAP
        shap.plots.waterfall(
            explanation,
            max_display=12,
            show=False
        )

        fig_waterfall = plt.gcf()

        st.pyplot(fig_waterfall)
        plt.close(fig_waterfall)

        # 6. Explication textuelle simple
        st.markdown("### Interprétation")

        positives = contributions[
            contributions > 0
        ].sort_values(ascending=False)

        negatives = contributions[
            contributions < 0
        ].sort_values()

        if not positives.empty:
            st.write(
                "**Principal facteur augmentant le risque :** "
                f"{positives.index[0]} "
                f"({positives.iloc[0]:+.3f})."
            )

        if not negatives.empty:
            st.write(
                "**Principal facteur diminuant le risque :** "
                f"{negatives.index[0]} "
                f"({negatives.iloc[0]:+.3f})."
            )

        st.caption(
            "Les contributions SHAP sont exprimées en "
            "log-odds. Elles décrivent les effets du modèle "
            "et ne constituent pas des relations causales."
        )

# ----------------------------------------------------------------------------
# Module 6 : API FastAPI
# ----------------------------------------------------------------------------
EDITABLE = {  # variable : (type, min, max)
    "Tuition fees up to date": ("bin", 0, 1),
    "Debtor": ("bin", 0, 1),
    "Scholarship holder": ("bin", 0, 1),
    "Age at enrollment": ("int", 17, 70),
    "Curricular units 1st sem (approved)": ("int", 0, 26),
    "Curricular units 1st sem (grade)": ("float", 0.0, 20.0),
    "Curricular units 2nd sem (approved)": ("int", 0, 26),
    "Curricular units 2nd sem (grade)": ("float", 0.0, 20.0),
}


def api_call(method, path, **kwargs):
    """Appelle l'API ; renvoie (données, message d'erreur)."""
    url = st.session_state.get("api_url", "http://127.0.0.1:8000").rstrip("/") + path
    try:
        r = requests.request(method, url, timeout=15, **kwargs)
    except requests.exceptions.RequestException:
        return None, (f"Impossible de joindre l'API à {url}. Lancez-la avec "
                      "« uvicorn api.main:app » et vérifiez l'adresse dans la barre latérale.")
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"L'API a répondu {r.status_code} : {detail}"
    return r.json(), None


def tab_api():
    st.subheader("Module 6 : API d'interprétation")
    st.caption("L'application appelle l'API FastAPI, qui sert les modèles entraînés dans le "
               "notebook. Documentation interactive de l'API : adresse de l'API + /docs.")

    if st.button("Tester la connexion", key="btn_api_ping"):
        data, err = api_call("GET", "/health")
        if err:
            st.error(err)
        else:
            st.success(f"API disponible : {len(data['models'])} modèles, "
                       f"{data['n_test']} étudiants de test.")
            st.session_state["api_models"] = data["models"]

    models = st.session_state.get("api_models")
    if not models:
        data, err = api_call("GET", "/models")
        if err:
            st.warning(err)
            return
        models = data["models"]
        st.session_state["api_models"] = models

    st.markdown("#### Prédiction pour un étudiant du jeu de test")
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Modèle", models, key="api_model")
    seuil = c2.slider("Seuil de décision", 0.01, 0.99, 0.30, 0.01, key="api_thr")
    idx = int(c3.number_input("Numéro de l'étudiant (0 à 884)", 0, 884, 0, key="api_idx"))

    student, err = api_call("GET", f"/students/{idx}")
    if err:
        st.error(err)
        return
    etu = dict(student["etudiant"])
    st.markdown("**Modifiez quelques caractéristiques** (les autres variables gardent la valeur "
                "de l'étudiant sélectionné) :")
    cols = st.columns(4)
    for i, (var, (kind, lo, hi)) in enumerate(EDITABLE.items()):
        key = f"edit_{var}_{idx}"
        with cols[i % 4]:
            if kind == "bin":
                etu[var] = float(st.selectbox(var, [0, 1], index=int(etu[var]), key=key))
            elif kind == "int":
                etu[var] = float(st.slider(var, lo, hi, int(min(max(etu[var], lo), hi)), key=key))
            else:
                etu[var] = float(st.slider(var, lo, hi, float(min(max(etu[var], lo), hi)),
                                           step=0.1, key=key))

    pred, err = api_call("POST", "/predict", json={"modele": model, "seuil": seuil, "etudiant": etu})
    if err:
        st.error(err)
        return
    p = pred["probabilite_decrochage"]
    g1, g2 = st.columns([1, 1])
    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": " %"},
            title={"text": "Probabilité de décrochage"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#355c7d"},
                   "threshold": {"line": {"color": "red", "width": 4}, "value": seuil * 100}}))
        fig.update_layout(height=300, margin=dict(t=60, b=10))
        st.plotly_chart(fig, width="stretch")
    with g2:
        if p >= seuil:
            st.error(f"Étudiant à risque (probabilité {p:.1%} ≥ seuil {seuil:.0%})")
        else:
            st.success(f"Étudiant sans alerte (probabilité {p:.1%} < seuil {seuil:.0%})")
        st.write("Situation réelle de cet étudiant : "
                 + ("**a décroché**" if student["decrochage_reel"] == 1 else "**n'a pas décroché**"))
        with st.expander("Voir les variables envoyées à l'API"):
            st.json(etu)

    st.markdown("#### Métriques calculées par l'API")
    met, err = api_call("GET", "/metrics", params={"modele": model, "seuil": seuil})
    if err:
        st.error(err)
        return
    m = st.columns(5)
    m[0].metric("AUC-ROC", f"{met['auc']:.3f}")
    m[1].metric("Précision", f"{met['precision']:.1%}")
    m[2].metric("Rappel", f"{met['rappel']:.1%}")
    m[3].metric("F1", f"{met['f1']:.3f}")
    m[4].metric("Accuracy", f"{met['accuracy']:.1%}")
    cm = met["matrice_confusion"]
    st.caption(f"Sur le jeu de test : {cm['vrais_positifs']} décrocheurs détectés, "
               f"{cm['faux_negatifs']} ratés, {cm['faux_positifs']} fausses alertes, "
               f"{cm['vrais_negatifs']} non-décrocheurs correctement écartés.")


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.title("Prédire le décrochage étudiant")
st.caption("Classification explicable : charger, préparer, comparer des modèles, puis expliquer "
           "leurs décisions. Data Mining, ENSI.")

progress_box = st.sidebar.container()
sidebar_api()
tabs = st.tabs(["1. Données", "2. Préparation", "3. Modélisation", "4. Évaluation",
                "5. Explicabilité", "6. API et prédiction"])
with tabs[0]:
    tab_data()
with tabs[1]:
    tab_prep()
with tabs[2]:
    tab_models()
with tabs[3]:
    tab_eval()
with tabs[4]:
    tab_xai()
with tabs[5]:
    tab_api()

sidebar_progress(progress_box)