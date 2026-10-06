"""Detecção de livros com HOG, janelas multiescala e classificadores treinados."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from src.detection.windows import DetectorConfig, window_boxes
from src.preprocess.hog import to_hog
from src.preprocess.image import normalize_size
from src.utils.coco import ImageRecord
from src.utils.geometry import clip_box, overlap_matrices


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    score: float


def _crop(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Recorta uma regiao delimitada por uma caixa.

    Args:
        image: Imagem de origem BGR.
        box: Caixa no formato ``x, y, largura, altura``.

    Returns:
        Recorte correspondente a caixa solicitada.
    """
    x, y, width, height = clip_box(box, image.shape[1], image.shape[0])
    return image[int(y) : int(np.ceil(y + height)), int(x) : int(np.ceil(x + width))]


def training_features(
    records: list[ImageRecord],
    negatives_per_image: int = 15,
    seed: int = 42,
    config: DetectorConfig | None = None,
    positives_per_book: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrai HOG de livros e de janelas de fundo com ocupação anotada mínima.

    Args:
        records: Imagens de treino e suas anotacoes COCO.
        negatives_per_image: Quantidade alvo de recortes negativos por imagem.
        seed: Semente para a amostragem aleatoria reprodutivel.
        config: Configuração de normalização e janelas estimada em train.
        positives_per_book: Máximo de janelas extras com IoU >= 0,60 por livro.

    Returns:
        Matriz de features HOG e vetor de rotulos binarios correspondentes.
    """
    if negatives_per_image < 1 or positives_per_book < 0:
        raise ValueError("negatives_per_image deve ser positivo e positives_per_book >= 0.")
    config = config or DetectorConfig()
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for record in records:
        image = cv2.imread(str(record.path))
        if image is None:
            raise ValueError(f"Imagem invalida: {record.path}")
        original_height, original_width = image.shape[:2]
        image, (scale_x, scale_y) = normalize_size(image, config.max_side)
        boxes = []
        for annotation in record.annotations:
            x, y, width, height = clip_box(annotation.bbox, original_width, original_height)
            if width > 0 and height > 0:
                boxes.append((x * scale_x, y * scale_y, width * scale_x, height * scale_y))
        candidates = np.asarray(
            list(window_boxes(image.shape[1], image.shape[0], config)), dtype=np.float64
        ).reshape(-1, 4)
        overlaps, occupancy = overlap_matrices(
            candidates, np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        )
        for box in boxes:
            crop = _crop(image, box)
            if crop.size:
                features.append(to_hog(crop))
                labels.append(1)
        # Usa também janelas que o detector realmente poderá encontrar.
        positive_indices: set[int] = set()
        for index in range(len(boxes)):
            valid = np.flatnonzero(overlaps[:, index] >= 0.6)
            ranked = valid[np.argsort(-overlaps[valid, index], kind="stable")]
            positive_indices.update(ranked[:positives_per_book].tolist())
        for index in sorted(positive_indices):
            features.append(to_hog(_crop(image, tuple(candidates[index]))))
            labels.append(1)

        # Uma janela dentro de um livro grande é rejeitada mesmo se sua IoU for baixa.
        negative_indices = np.flatnonzero(occupancy.sum(axis=1) <= 0.01)
        selected = rng.choice(
            negative_indices, size=min(negatives_per_image, len(negative_indices)), replace=False
        )
        for index in selected:
            features.append(to_hog(_crop(image, tuple(candidates[index]))))
            labels.append(0)
    if len(set(labels)) < 2:
        raise ValueError("O treinamento precisa de exemplos válidos de livro e fundo.")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def non_maximum_suppression(
    detections: list[Detection], threshold: float = 0.35
) -> list[Detection]:
    """Remove deteccoes redundantes com alta sobreposicao.

    Args:
        detections: Caixas candidatas com suas pontuacoes.
        threshold: IoU maximo permitido entre caixas mantidas.

    Returns:
        Deteccoes filtradas por supressao nao maxima.
    """
    if not detections:
        return []
    boxes = np.asarray([item.bbox for item in detections], dtype=np.float64)
    order = np.argsort(-np.asarray([item.score for item in detections]), kind="stable")
    kept = []
    while order.size:
        index = int(order[0])
        kept.append(detections[index])
        remaining = order[1:]
        overlaps, _ = overlap_matrices(boxes[remaining], boxes[index : index + 1])
        order = remaining[overlaps[:, 0] < threshold]
    return kept


def detect(
    image: np.ndarray,
    classifier: object,
    score_threshold: float = 0.0,
    step: int = 32,
    batch_size: int = 256,
) -> list[Detection]:
    """Varre a imagem em multiplas escalas e retorna caixas apos NMS.

    Args:
        image: Imagem BGR em que os livros serao procurados.
        classifier: Classificador previamente treinado.
        score_threshold: Pontuacao minima para aceitar uma janela candidata.
        step: Passo, em pixels, entre janelas consecutivas.
        batch_size: Quantidade máxima de descritores enviada por previsão em lote.

    Returns:
        Deteccoes finais apos supressao nao maxima.
    """
    return detect_many(image, {"model": classifier}, {"model": score_threshold}, step, batch_size)[
        "model"
    ]


def score_features(classifier: object, features: np.ndarray) -> np.ndarray:
    """Calcula pontuações na escala nativa de cada classificador binário.

    Args:
        classifier: Pipeline treinado para as classes zero e um.
        features: Matriz de descritores HOG.

    Returns:
        Margens da regressão ou probabilidades do MLP menos 0,5.
    """
    if hasattr(classifier, "decision_function"):
        return np.asarray(classifier.decision_function(features))
    positive_class = int(np.flatnonzero(np.asarray(classifier.classes_) == 1)[0])
    return np.asarray(classifier.predict_proba(features)[:, positive_class] - 0.5)


def detect_many(
    image: np.ndarray,
    classifiers: Mapping[str, object],
    score_thresholds: Mapping[str, float],
    step: int = 32,
    batch_size: int = 256,
) -> dict[str, list[Detection]]:
    """Compartilha a extração HOG entre modelos com as mesmas janelas.

    Args:
        image: Imagem BGR a ser analisada.
        classifiers: Pipelines treinados, identificados pelo nome do experimento.
        score_thresholds: Pontuação mínima de cada experimento.
        step: Passo máximo entre janelas.
        batch_size: Quantidade de janelas por lote de descritores.

    Returns:
        Detecções após NMS para cada experimento, nas coordenadas originais.
    """
    if not classifiers or classifiers.keys() != score_thresholds.keys():
        raise ValueError("Informe os mesmos nomes para modelos e limiares.")
    if batch_size < 1:
        raise ValueError("batch_size deve ser positivo.")
    configs = [
        getattr(model, "detector_config_", DetectorConfig()) for model in classifiers.values()
    ]
    config = configs[0]
    if any(other != config for other in configs[1:]):
        raise ValueError("Os modelos devem compartilhar a configuração de janelas.")
    normalized, (scale_x, scale_y) = normalize_size(image, config.max_side)
    boxes = list(window_boxes(normalized.shape[1], normalized.shape[0], config, step))
    candidates: dict[str, list[Detection]] = {name: [] for name in classifiers}
    for start in range(0, len(boxes), batch_size):
        batch = boxes[start : start + batch_size]
        features = np.asarray([to_hog(_crop(normalized, box)) for box in batch], dtype=np.float32)
        for name, classifier in classifiers.items():
            scores = score_features(classifier, features)
            for box, score in zip(batch, scores, strict=True):
                if score >= score_thresholds[name]:
                    x, y, width, height = box
                    left, top = round(x / scale_x), round(y / scale_y)
                    right, bottom = round((x + width) / scale_x), round((y + height) / scale_y)
                    candidates[name].append(
                        Detection((left, top, right - left, bottom - top), float(score))
                    )
    return {name: non_maximum_suppression(items) for name, items in candidates.items()}
