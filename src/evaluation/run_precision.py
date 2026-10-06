"""Experimento curto e rastreável: regularização, negativos difíceis e MLP."""

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from time import perf_counter

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from src.detection.classifiers import train_classifier
from src.detection.hard_negatives import mine_hard_negatives
from src.detection.hog_detector import training_features
from src.evaluation.run_logistic import report_progress
from src.evaluation.thresholds import (
    evaluate_models_thresholds,
    save_threshold_report,
    select_threshold,
)
from src.utils.coco import load_coco_split
from src.utils.model_io import load_model, save_model
from src.utils.path import EVALUATION_DIR

LOGISTIC_THRESHOLDS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]
# A pontuação do MLP é p(livro) - 0,5, e não a margem da regressão.
MLP_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.49, 0.495, 0.499]


def main() -> None:
    """Executa variantes controladas em train e valida uma única vez por imagem.

    Args:
        Nenhum; o protocolo limitado é definido neste módulo.

    Returns:
        Nenhum; salva pickles versionados, CSVs e um manifesto JSON.
    """
    started = perf_counter()
    cv2.setNumThreads(1)
    with threadpool_limits(limits=2):
        train_records = load_coco_split("train")
        valid_records = load_coco_split("valid")
        baseline = load_model("logistic_regression_002")
        config = baseline.detector_config_
        features, labels = training_features(train_records, negatives_per_image=30, config=config)
        print(f"Exemplos comuns: {features.shape}", flush=True)
        print("Treinando regressão com C=0,01...", flush=True)
        regularized = train_classifier(
            "logistic_regression", features, labels, config, regularization=0.01
        )
        print("Treinando MLP sobre os mesmos recortes...", flush=True)
        mlp = train_classifier("mlp", features, labels, config)
        print("Minerando até 8 negativos em 30 imagens de train...", flush=True)
        mined = mine_hard_negatives(train_records, baseline, progress=report_progress)
        print(f"Negativos difíceis adicionados: {len(mined.features)}", flush=True)
        augmented_features = np.concatenate([features, mined.features])
        augmented_labels = np.concatenate([labels, np.zeros(len(mined.features), dtype=np.int32)])
        hard_model = train_classifier(
            "logistic_regression", augmented_features, augmented_labels, config
        )
        models = {
            "Regressão — 30 negativos": baseline,
            "Regressão — C=0,01": regularized,
            "Regressão — negativos difíceis": hard_model,
            "MLP — 30 negativos": mlp,
        }
        grades = {name: LOGISTIC_THRESHOLDS for name in models}
        grades["MLP — 30 negativos"] = MLP_THRESHOLDS
        print("Validando todos os modelos com HOG compartilhado...", flush=True)
        results = evaluate_models_thresholds(
            valid_records, models, grades, progress=report_progress
        )
        entries = []
        for name, model in models.items():
            selected = None
            try:
                selected = select_threshold(results[name], minimum_recall=0.5)
            except ValueError as error:
                print(f"{name}: {error}", flush=True)
            if selected is not None:
                model.score_threshold_ = selected.threshold
            else:
                # Não deixa um limiar antigo parecer selecionado nesta grade.
                model.__dict__.pop("score_threshold_", None)
            model.training_metadata_ = {
                "negatives_per_image": 30,
                "positives_per_book": 2,
                "seed": 42,
                "samples": len(augmented_labels) if model is hard_model else len(labels),
                "hard_negatives": len(mined.features) if model is hard_model else 0,
                "hard_negative_images": list(mined.image_ids) if model is hard_model else [],
                "hard_negative_boxes": list(mined.source_boxes) if model is hard_model else [],
                "regularization_c": 0.01 if model is regularized else 1.0,
                "minimum_recall": 0.5,
                "trained_split": "train",
                "threshold_split": "valid",
                "threshold_selected": selected is not None,
                "experiment": name,
            }
            path = save_model(model, "mlp" if model is mlp else "logistic_regression")
            report = save_threshold_report(results[name], path.stem)
            entry = {
                "label": name,
                "model": path.stem,
                "report": report.name,
                "selected": asdict(selected) if selected is not None else None,
                "metadata": model.training_metadata_,
            }
            entries.append(entry)
            print(f"{name}: {selected}\nSalvo: {path.name}", flush=True)
        eligible = [item for item in entries if item["selected"] is not None]
        winner = max(
            eligible,
            key=lambda item: (
                item["selected"]["metrics"]["precision"],
                item["selected"]["metrics"]["f1_score"],
            ),
            default=None,
        )
        manifest = {
            "date": datetime.now(timezone(timedelta(hours=-3))).date().isoformat(),
            "split": "valid",
            "images": len(valid_records),
            "iou_threshold": 0.5,
            "minimum_recall": 0.5,
            "elapsed_seconds": round(perf_counter() - started, 1),
            "recommended_model": winner["model"] if winner else None,
            "experiments": entries,
        }
        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
        version = 1
        while True:
            destination = EVALUATION_DIR / f"precision_run_{version:03d}.json"
            try:
                file = destination.open("x", encoding="utf-8")
            except FileExistsError:
                version += 1
                continue
            with file:
                json.dump(manifest, file, ensure_ascii=False, indent=2)
            break
        print(f"Manifesto: {destination}\nRecomendado: {manifest['recommended_model']}", flush=True)
        print(f"Tempo total: {manifest['elapsed_seconds']} s", flush=True)


if __name__ == "__main__":
    main()
