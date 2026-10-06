"""Classificadores clássicos comparados sobre as mesmas features HOG."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.detection.windows import DetectorConfig

ClassifierName = Literal["logistic_regression", "mlp"]


def build_classifier(name: ClassifierName, regularization: float = 1.0) -> Pipeline:
    """Cria um classificador com hiperparametros iniciais reproduziveis.

    Args:
        name: Identificador do classificador a ser instanciado.
        regularization: Parâmetro C da regressão; valores menores regularizam mais.

    Returns:
        Classificador ainda nao treinado.
    """
    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError("C deve ser positivo e finito.")
    models: dict[ClassifierName, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=regularization, max_iter=3000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=1000,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    return models[name]


def train_classifier(
    name: ClassifierName,
    features: np.ndarray,
    labels: np.ndarray,
    config: DetectorConfig | None = None,
    *,
    regularization: float = 1.0,
) -> Pipeline:
    """Treina um classificador sobre um conjunto de features HOG.

    Args:
        name: Identificador do classificador a treinar.
        features: Matriz de descritores HOG de treino.
        labels: Rotulos binarios associados a cada descritor.
        config: Configuração de busca usada para extrair os exemplos de treino.
        regularization: Parâmetro C da regressão; ignorado pelo MLP.

    Returns:
        Classificador treinado.
    """
    classifier = build_classifier(name, regularization)
    classifier.fit(features, labels)
    classifier.detector_config_ = config or DetectorConfig()
    return classifier
