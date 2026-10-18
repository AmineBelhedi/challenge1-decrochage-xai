
"""Moteur d'explicabilité SHAP du modèle XGBoost."""

import numpy as np
import pandas as pd
import shap


class ShapExplainer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.prep = pipeline.named_steps["prep"]
        self.model = pipeline.named_steps["clf"]

        self.features = list(pipeline.feature_names_in_)
        self.encoded_names = list(
            self.prep.get_feature_names_out()
        )

        # Identifier les variables catégorielles
        self.categorical_cols = next(
            list(columns)
            for name, _, columns in self.prep.transformers_
            if name == "cat"
        )

        self.categorical_cols.sort(
            key=len, reverse=True
        )

        # Construire une fois la matrice de regroupement
        self.M = self._build_grouping_matrix()

        # Initialiser SHAP une seule fois
        self.explainer = shap.TreeExplainer(self.model)

    def _original_feature(self, encoded_name):
        kind, name = encoded_name.split("__", 1)

        if kind == "cat":
            for col in self.categorical_cols:
                if name.startswith(col + "_"):
                    return col

            raise ValueError(
                f"Variable catégorielle inconnue : {encoded_name}"
            )

        if name not in self.features:
            raise ValueError(
                f"Variable inconnue : {encoded_name}"
            )

        return name

    def _build_grouping_matrix(self):
        M = np.zeros((
            len(self.encoded_names),
            len(self.features)
        ))

        for i, name in enumerate(self.encoded_names):
            origin = self._original_feature(name)
            j = self.features.index(origin)
            M[i, j] = 1

        assert np.all(M.sum(axis=1) == 1)
        return M

    def explain_student(self, student):
        """Expliquer un étudiant avec SHAP."""

        # Accepte un dictionnaire de 36 variables
        row = pd.DataFrame(
            [student],
            columns=self.features
        )

        # Appliquer le prétraitement
        transformed = self.prep.transform(row)

        # Calculer les contributions SHAP
        values = np.asarray(
            self.explainer.shap_values(transformed)
        )

        # Regrouper les 235 contributions en 36
        grouped = values @ self.M

        # Calculer la probabilité
        probability = float(
            self.pipeline.predict_proba(row)[0, 1]
        )

        # Contributions triées par importance absolue
        contributions = pd.Series(
            grouped[0],
            index=self.features
        )

        contributions = contributions.reindex(
            contributions.abs().sort_values(
                ascending=False
            ).index
        )

        return {
            "probabilite": probability,
            "base_value": float(
                np.ravel(self.explainer.expected_value)[0]
            ),
            "contributions": {
                name: float(value)
                for name, value in contributions.items()
            },
            "unite_shap": "log_odds"
        }
