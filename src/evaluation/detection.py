from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.detection.hog_detector import Detection, detect
from src.detection.windows import DetectorConfig, window_boxes
from src.utils.coco import ImageRecord
from src.utils.geometry import clip_box, iou, overlap_matrices


@dataclass(frozen=True)
class CoverageMetrics:
    """Cobertura geométrica das janelas, independente do classificador."""

    total: int
    covered: int
    recall_upper_bound: float
    total_windows: int


def candidate_coverage(
    records: list[ImageRecord], config: DetectorConfig, iou_threshold: float = 0.5
) -> CoverageMetrics:
    """Mede quantos livros podem ser localizados pelas janelas geradas.

    Args:
        records: Imagens anotadas de uma divisão do dataset.
        config: Configuração das janelas, estimada somente em train.
        iou_threshold: IoU mínima para considerar uma anotação alcançável.

    Returns:
        Cobertura das anotações e limite superior geométrico da revocação.
    """
    total = covered = total_windows = 0
    for record in records:
        scale = config.max_side / max(record.width, record.height)
        width, height = max(1, round(record.width * scale)), max(1, round(record.height * scale))
        boxes = []
        for annotation in record.annotations:
            x, y, w, h = clip_box(annotation.bbox, record.width, record.height)
            if w > 0 and h > 0:
                boxes.append(
                    (
                        x * width / record.width,
                        y * height / record.height,
                        w * width / record.width,
                        h * height / record.height,
                    )
                )
        windows = np.asarray(list(window_boxes(width, height, config)), dtype=np.float64).reshape(
            -1, 4
        )
        total_windows += len(windows)
        if boxes:
            overlaps, _ = overlap_matrices(windows, np.asarray(boxes, dtype=np.float64))
            total += len(boxes)
            covered += int((overlaps.max(axis=0) >= iou_threshold).sum())
    return CoverageMetrics(total, covered, covered / total if total else 0.0, total_windows)


@dataclass(frozen=True)
class DetectionMetrics:
    """Metricas agregadas de deteccao de objetos."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


def count_matches(
    predictions: list[Detection],
    ground_truths: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Compara deteccoes e caixas reais por correspondencia um-a-um.

    Args:
        predictions: Deteccoes previstas, ordenadas ou nao por pontuacao.
        ground_truths: Caixas reais no formato ``x, y, largura, altura``.
        iou_threshold: IoU minimo para considerar uma deteccao correta.

    Returns:
        Quantidade de verdadeiros positivos, falsos positivos e falsos negativos.
    """
    unmatched = set(range(len(ground_truths)))
    true_positives = 0
    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        matches = [
            index
            for index in unmatched
            if iou(prediction.bbox, ground_truths[index]) >= iou_threshold
        ]
        if matches:
            best_match = max(matches, key=lambda index: iou(prediction.bbox, ground_truths[index]))
            unmatched.remove(best_match)
            true_positives += 1
    false_positives = len(predictions) - true_positives
    false_negatives = len(unmatched)
    return true_positives, false_positives, false_negatives


def calculate_metrics(
    true_positives: int, false_positives: int, false_negatives: int
) -> DetectionMetrics:
    """Calcula precisao, revocacao e F1 a partir das contagens de deteccao.

    Args:
        true_positives: Numero de livros corretamente detectados.
        false_positives: Numero de caixas previstas sem livro correspondente.
        false_negatives: Numero de livros reais nao detectados.

    Returns:
        Estrutura com contagens e metricas agregadas entre zero e um.
    """
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionMetrics(
        true_positives, false_positives, false_negatives, precision, recall, f1_score
    )


def evaluate_detector(
    records: list[ImageRecord],
    classifier: object,
    score_threshold: float = 0.0,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Avalia o detector em uma divisao anotada do dataset.

    Args:
        records: Imagens e caixas reais da divisao a ser avaliada.
        classifier: Classificador previamente treinado.
        score_threshold: Pontuacao minima para uma janela ser considerada deteccao.
        iou_threshold: IoU minimo para contabilizar um verdadeiro positivo.

    Returns:
        Metricas agregadas para todas as imagens avaliadas.
    """
    true_positives = false_positives = false_negatives = 0
    for record in records:
        image = cv2.imread(str(record.path))
        if image is None:
            raise ValueError(f"Imagem invalida: {record.path}")
        predictions = detect(image, classifier, score_threshold=score_threshold)
        ground_truths = [
            clip_box(annotation.bbox, image.shape[1], image.shape[0])
            for annotation in record.annotations
        ]
        ground_truths = [box for box in ground_truths if box[2] > 0 and box[3] > 0]
        tp, fp, fn = count_matches(predictions, ground_truths, iou_threshold)
        true_positives += tp
        false_positives += fp
        false_negatives += fn
    return calculate_metrics(true_positives, false_positives, false_negatives)
