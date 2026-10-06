"""Mineração limitada de falsos alarmes usando exclusivamente imagens de train."""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from src.detection.hog_detector import _crop, detect
from src.detection.windows import DetectorConfig
from src.preprocess.hog import to_hog
from src.preprocess.image import normalize_size
from src.utils.coco import ImageRecord
from src.utils.geometry import clip_box, overlap_matrices


@dataclass(frozen=True)
class MiningResult:
    """Descritores e rastreabilidade dos negativos difíceis selecionados."""

    features: np.ndarray
    image_ids: tuple[int, ...]
    source_boxes: tuple[tuple[int, tuple[int, int, int, int]], ...]


def mine_hard_negatives(
    records: list[ImageRecord],
    classifier: object,
    max_images: int = 30,
    per_image: int = 8,
    seed: int = 42,
    progress: Callable[[int, int], None] | None = None,
) -> MiningResult:
    """Seleciona fundos confundidos com livros e exclui fragmentos anotados.

    Args:
        records: Registros exclusivamente da divisão train.
        classifier: Modelo de referência treinado em train.
        max_images: Limite de imagens, amostradas sem reposição para reduzir o custo.
        per_image: Máximo de falsos alarmes distintos por imagem, após NMS.
        seed: Semente reproduzível para a escolha das imagens.
        progress: Callback opcional com imagens processadas e total.

    Returns:
        Descritores das janelas com ocupação anotada total de no máximo 1%.
    """
    if max_images < 1 or per_image < 1 or not records:
        raise ValueError("Informe imagens e limites positivos para a mineração.")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), min(max_images, len(records)), replace=False)
    config = getattr(classifier, "detector_config_", DetectorConfig())
    features, sources, image_ids = [], [], []
    for completed, index in enumerate(indices, start=1):
        record = records[int(index)]
        image = cv2.imread(str(record.path))
        if image is None:
            raise ValueError(f"Imagem inválida: {record.path}")
        image_ids.append(record.id)
        candidates = detect(image, classifier, score_threshold=0.0)
        targets = np.asarray(
            [clip_box(a.bbox, image.shape[1], image.shape[0]) for a in record.annotations],
            dtype=np.float64,
        ).reshape(-1, 4)
        boxes = np.asarray([d.bbox for d in candidates], dtype=np.float64).reshape(-1, 4)
        _, occupancy = overlap_matrices(boxes, targets)
        eligible = np.flatnonzero(occupancy.sum(axis=1) <= 0.01)[:per_image]
        normalized, (scale_x, scale_y) = normalize_size(image, config.max_side)
        for candidate_index in eligible:
            box = candidates[int(candidate_index)].bbox
            x, y, width, height = box
            crop = _crop(normalized, (x * scale_x, y * scale_y, width * scale_x, height * scale_y))
            features.append(to_hog(crop))
            sources.append((record.id, box))
        if progress is not None:
            progress(completed, len(indices))
    dimension = len(to_hog(np.zeros((128, 96, 3), dtype=np.uint8)))
    return MiningResult(
        np.asarray(features, dtype=np.float32).reshape(-1, dimension),
        tuple(image_ids),
        tuple(sources),
    )
