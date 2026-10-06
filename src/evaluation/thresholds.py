"""Comparação de limiares de detecção com uma única passagem pelas imagens."""

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.detection.hog_detector import detect_many
from src.evaluation.detection import DetectionMetrics, calculate_metrics, count_matches
from src.utils.coco import ImageRecord
from src.utils.geometry import clip_box
from src.utils.path import EVALUATION_DIR


@dataclass(frozen=True)
class ThresholdResult:
    """Limiar avaliado e métricas agregadas de detecção."""

    threshold: float
    metrics: DetectionMetrics


def evaluate_thresholds(
    records: list[ImageRecord],
    classifier: object,
    thresholds: list[float],
    iou_threshold: float = 0.5,
    progress: Callable[[int, int], None] | None = None,
) -> list[ThresholdResult]:
    """Compara limiares em validação sem repetir HOG ou previsões.

    O NMS guloso processa caixas por pontuação decrescente. Caixas com pontuação
    inferior não suprimem caixas superiores; por isso, filtrar sua saída pelo
    limiar é equivalente a executar esse mesmo NMS separadamente em cada limiar.

    Args:
        records: Registros da divisão valid.
        classifier: Classificador já treinado somente em train.
        thresholds: Limiares finitos a comparar, na escala de pontuação do modelo.
        iou_threshold: IoU mínima para contabilizar um verdadeiro positivo.
        progress: Callback opcional com quantidades de imagens concluídas e totais.

    Returns:
        Métricas de detecção para cada limiar, em ordem crescente.
    """
    return evaluate_models_thresholds(
        records, {"model": classifier}, {"model": thresholds}, iou_threshold, progress
    )["model"]


def evaluate_models_thresholds(
    records: list[ImageRecord],
    classifiers: Mapping[str, object],
    thresholds: Mapping[str, list[float]],
    iou_threshold: float = 0.5,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[ThresholdResult]]:
    """Compara modelos e limiares reutilizando HOG por imagem e por lote.

    Args:
        records: Imagens de validação; não forneça test durante ajustes.
        classifiers: Pipelines com a mesma configuração de janelas.
        thresholds: Grade na escala própria de cada modelo.
        iou_threshold: IoU mínima para correspondência um-a-um.
        progress: Callback opcional com imagens concluídas e total.

    Returns:
        Tabelas de métricas agregadas por modelo e limiar.
    """
    if not records or not classifiers or classifiers.keys() != thresholds.keys():
        raise ValueError("Informe imagens e nomes correspondentes para modelos e limiares.")
    values = {name: sorted(set(float(v) for v in grade)) for name, grade in thresholds.items()}
    if any(not grade or not np.isfinite(grade).all() for grade in values.values()):
        raise ValueError("Informe uma lista de limiares finitos para cada modelo.")
    if not 0 < iou_threshold <= 1:
        raise ValueError("A IoU deve estar no intervalo (0, 1].")
    totals = {name: {value: [0, 0, 0] for value in grade} for name, grade in values.items()}
    for completed, record in enumerate(records, start=1):
        image = cv2.imread(str(record.path))
        if image is None:
            raise ValueError(f"Imagem inválida: {record.path}")
        predictions = detect_many(
            image, classifiers, {name: grade[0] for name, grade in values.items()}
        )
        targets = [clip_box(a.bbox, image.shape[1], image.shape[0]) for a in record.annotations]
        targets = [box for box in targets if box[2] > 0 and box[3] > 0]
        for name, grade in values.items():
            for value in grade:
                selected = [item for item in predictions[name] if item.score >= value]
                counts = count_matches(selected, targets, iou_threshold)
                totals[name][value] = [
                    a + b for a, b in zip(totals[name][value], counts, strict=True)
                ]
        if progress is not None:
            progress(completed, len(records))
    return {
        name: [ThresholdResult(value, calculate_metrics(*totals[name][value])) for value in grade]
        for name, grade in values.items()
    }


def select_threshold(
    results: list[ThresholdResult], minimum_recall: float = 0.5
) -> ThresholdResult:
    """Escolhe a maior precisão entre limiares que mantêm a revocação mínima.

    Args:
        results: Resultados obtidos em valid.
        minimum_recall: Fração mínima de livros que precisam ser detectados.

    Returns:
        Resultado selecionado, com desempate por F1 e menor limiar.
    """
    if not 0 <= minimum_recall <= 1:
        raise ValueError("minimum_recall deve estar entre zero e um.")
    eligible = [
        item
        for item in results
        if item.metrics.recall >= minimum_recall and item.metrics.true_positives > 0
    ]
    if not eligible:
        raise ValueError("Nenhum limiar atende à revocação mínima; examine a tabela de resultados.")
    return max(
        eligible, key=lambda item: (item.metrics.precision, item.metrics.f1_score, -item.threshold)
    )


def save_threshold_report(results: list[ThresholdResult], model_name: str) -> Path:
    """Salva a tabela de limiares sem substituir relatórios anteriores.

    Args:
        results: Resultados completos da comparação em valid.
        model_name: Nome do pickle relacionado ao experimento, sem extensão.

    Returns:
        Caminho do novo relatório CSV, com sufixo se necessário.
    """
    if not model_name or any(char in model_name for char in '/\\:*?"<>|'):
        raise ValueError("Informe um nome de modelo válido, sem diretórios.")
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    version = 0
    while True:
        suffix = f"_{version:03d}" if version else ""
        destination = EVALUATION_DIR / f"{model_name}_thresholds{suffix}.csv"
        try:
            file = destination.open("x", encoding="utf-8", newline="")
        except FileExistsError:
            version += 1
            continue
        with file:
            writer = csv.writer(file)
            writer.writerow(["threshold", "tp", "fp", "fn", "precision", "recall", "f1_score"])
            for item in results:
                metrics = item.metrics
                writer.writerow(
                    [
                        item.threshold,
                        metrics.true_positives,
                        metrics.false_positives,
                        metrics.false_negatives,
                        metrics.precision,
                        metrics.recall,
                        metrics.f1_score,
                    ]
                )
        return destination


def load_threshold_report(report: Path) -> list[ThresholdResult]:
    """Lê resultados já calculados sem repetir treinamento ou detecção.

    Args:
        report: Caminho do CSV exportado pelo experimento.

    Returns:
        Resultados de cada limiar com as métricas registradas.
    """
    with report.open(encoding="utf-8", newline="") as file:
        return [
            ThresholdResult(
                float(row["threshold"]),
                DetectionMetrics(
                    int(row["tp"]),
                    int(row["fp"]),
                    int(row["fn"]),
                    float(row["precision"]),
                    float(row["recall"]),
                    float(row["f1_score"]),
                ),
            )
            for row in csv.DictReader(file)
        ]
