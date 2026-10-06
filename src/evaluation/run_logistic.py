"""Experimento reproduzível de Regressão Logística com comparação de limiares."""

import argparse
from time import perf_counter

import cv2

from src.detection.classifiers import train_classifier
from src.detection.hog_detector import training_features
from src.detection.windows import fit_window_config
from src.evaluation.thresholds import evaluate_thresholds, save_threshold_report, select_threshold
from src.utils.coco import load_coco_split
from src.utils.model_io import save_model


def report_progress(completed: int, total: int) -> None:
    """Mostra o andamento sem repetir a extração das características por limiar.

    Args:
        completed: Quantidade de imagens avaliadas.
        total: Quantidade total de imagens de validação.

    Returns:
        Nenhum.
    """
    if completed % 5 == 0 or completed == total:
        print(f"Validação: {completed}/{total} imagens", flush=True)


def main() -> None:
    """Treina uma regressão, avalia limiares em valid e salva os resultados.

    Args:
        Nenhum; os parâmetros são lidos da linha de comando.

    Returns:
        Nenhum.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negatives", type=int, default=15)
    parser.add_argument("--minimum-recall", type=float, default=0.5)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    )
    args = parser.parse_args()
    # Recortes HOG pequenos não se beneficiam de criar trabalho paralelo a cada janela.
    cv2.setNumThreads(1)
    started = perf_counter()
    train_records = load_coco_split("train")
    valid_records = load_coco_split("valid")
    config = fit_window_config(train_records)
    features, labels = training_features(
        train_records, negatives_per_image=args.negatives, config=config
    )
    print(f"Treinamento: {features.shape}", flush=True)
    classifier = train_classifier("logistic_regression", features, labels, config)
    results = evaluate_thresholds(
        valid_records, classifier, args.thresholds, progress=report_progress
    )
    print("limiar,tp,fp,fn,precision,recall,f1", flush=True)
    for item in results:
        m = item.metrics
        print(
            f"{item.threshold},{m.true_positives},{m.false_positives},{m.false_negatives},"
            f"{m.precision:.6f},{m.recall:.6f},{m.f1_score:.6f}",
            flush=True,
        )
    try:
        selected = select_threshold(results, args.minimum_recall)
    except ValueError as error:
        selected = None
        print(f"Limiar não selecionado: {error}", flush=True)
    if selected is not None:
        classifier.score_threshold_ = selected.threshold
    classifier.training_metadata_ = {
        "negatives_per_image": args.negatives,
        "positives_per_book": 2,
        "seed": 42,
        "samples": len(labels),
        "minimum_recall": args.minimum_recall,
        "trained_split": "train",
        "threshold_split": "valid",
        "threshold_selected": selected is not None,
    }
    model_file = save_model(classifier, "logistic_regression")
    report_file = save_threshold_report(results, model_file.stem)
    print(f"Selecionado: {selected}", flush=True)
    print(f"Modelo: {model_file}", flush=True)
    print(f"Relatório: {report_file}", flush=True)
    print(f"Tempo total: {perf_counter() - started:.1f} segundos", flush=True)


if __name__ == "__main__":
    main()
