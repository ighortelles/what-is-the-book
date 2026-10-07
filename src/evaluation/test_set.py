"""Avaliação final em test com pesos e limiares definidos exclusivamente em valid."""

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import cv2
from threadpoolctl import threadpool_limits

from src.evaluation.thresholds import evaluate_models_thresholds
from src.utils.coco import ImageRecord, load_coco_split
from src.utils.model_io import load_model
from src.utils.path import EVALUATION_DIR, model_path
from src.utils.presentation import latest_manifest


def dataset_fingerprint(records: list[ImageRecord]) -> str:
    """Identifica imagens e anotações para não reutilizar resultados de outro dataset.

    Args:
        records: Registros completos da divisão test, sem amostragem.

    Returns:
        SHA-256 das imagens, dimensões e caixas anotadas, na ordem de avaliação.
    """
    digest = sha256()
    for record in records:
        description = {
            "id": record.id,
            "file": record.path.name,
            "width": record.width,
            "height": record.height,
            "boxes": [annotation.bbox for annotation in record.annotations],
            "image_sha256": sha256(record.path.read_bytes()).hexdigest(),
        }
        digest.update(json.dumps(description, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def save_test_report(report: dict[str, object]) -> Path:
    """Salva uma avaliação final sem substituir relatórios anteriores.

    Args:
        report: Protocolo congelado, métricas de todos os modelos e tempo total.

    Returns:
        Caminho do JSON criado com versão numérica exclusiva.
    """
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        destination = EVALUATION_DIR / f"test_evaluation_{version:03d}.json"
        try:
            file = destination.open("x", encoding="utf-8")
        except FileExistsError:
            version += 1
            continue
        with file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        return destination


def evaluate_test_models(
    validation_manifest: dict | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, dict]:
    """Avalia os snapshots congelados ou lê a avaliação idêntica já salva.

    Args:
        validation_manifest: Manifesto de seleção em valid; usa o mais recente por padrão.
        progress: Callback opcional com quantidade de imagens de test concluídas e total.

    Returns:
        Caminho e conteúdo do relatório, incluindo métricas, limiares e hashes dos pesos.
    """
    manifest = validation_manifest if validation_manifest is not None else latest_manifest()
    if manifest["split"] != "valid":
        raise ValueError("Os modelos e limiares devem ter sido selecionados em valid.")
    records = load_coco_split("test")
    snapshots = []
    for entry in manifest["experiments"]:
        if entry["selected"] is None:
            raise ValueError(
                f"Defina um limiar em valid antes de avaliar {entry['label']} em test."
            )
        path = model_path(entry["model"])
        snapshots.append(
            {
                "name": entry["model"],
                "label": entry["label"],
                "threshold": float(entry["selected"]["threshold"]),
                "model_sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not snapshots or len({item["name"] for item in snapshots}) != len(snapshots):
        raise ValueError("Informe pelo menos um modelo e arquivos distintos por experimento.")
    protocol = {
        "split": "test",
        "threshold_source": "valid",
        "iou_threshold": float(manifest["iou_threshold"]),
        "step": 32,
        "recommended_model_from_valid": manifest["recommended_model"],
        "images": len(records),
        "annotations": sum(len(record.annotations) for record in records),
        "dataset_sha256": dataset_fingerprint(records),
        "models": snapshots,
    }
    for candidate in sorted(EVALUATION_DIR.glob("test_evaluation_*.json"), reverse=True):
        with candidate.open(encoding="utf-8") as file:
            previous = json.load(file)
        if previous.get("protocol") == protocol:
            return candidate, previous

    # Não treina, não minera negativos e não procura novos limiares no conjunto test.
    started = perf_counter()
    classifiers = {item["name"]: load_model(item["name"]) for item in snapshots}
    grades = {item["name"]: [item["threshold"]] for item in snapshots}
    with threadpool_limits(limits=2):
        evaluated = evaluate_models_thresholds(
            records, classifiers, grades, protocol["iou_threshold"], progress
        )
    report = {
        "date": datetime.now(timezone(timedelta(hours=-3))).date().isoformat(),
        "protocol": protocol,
        "elapsed_seconds": round(perf_counter() - started, 2),
        "results": [
            {
                "model": item["name"],
                "label": item["label"],
                "threshold": item["threshold"],
                "metrics": asdict(evaluated[item["name"]][0].metrics),
            }
            for item in snapshots
        ],
    }
    return save_test_report(report), report


def report_test_progress(completed: int, total: int) -> None:
    """Exibe o progresso identificando explicitamente a divisão de teste.

    Args:
        completed: Quantidade de imagens de test concluídas.
        total: Quantidade total de imagens de test.

    Returns:
        Nenhum.
    """
    if completed % 5 == 0 or completed == total:
        print(f"Teste: {completed}/{total} imagens", flush=True)


def main() -> None:
    """Executa a avaliação final reproduzível dos modelos congelados.

    Args:
        Nenhum.

    Returns:
        Nenhum; exibe as métricas e salva ou reutiliza o relatório correspondente.
    """
    cv2.setNumThreads(1)
    print("Test: limiares congelados em valid, sem novos ajustes.", flush=True)
    path, report = evaluate_test_models(progress=report_test_progress)
    for result in report["results"]:
        print(result, flush=True)
    print(f"Relatório: {path}\nTempo da avaliação: {report['elapsed_seconds']} s", flush=True)


if __name__ == "__main__":
    main()
