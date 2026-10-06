"""Gráfico reproduzível a partir de um relatório de limiares já calculado."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def plot_threshold_report(report: Path, selected_threshold: float) -> Path:
    """Exporta precisão, revocação e falsos positivos sem executar o detector.

    Args:
        report: Caminho do CSV gerado por save_threshold_report.
        selected_threshold: Limiar escolhido para indicar no gráfico.

    Returns:
        Caminho da imagem PNG gerada ao lado do relatório.
    """
    with report.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    thresholds = [float(row["threshold"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    values = (
        [100 * float(row["precision"]) for row in rows],
        [100 * float(row["recall"]) for row in rows],
        [int(row["fp"]) for row in rows],
    )
    labels = ("Precisão (%)", "Revocação (%)", "Falsos positivos")
    for axis, series, label in zip(axes, values, labels, strict=True):
        axis.plot(thresholds, series, "o-", color="#245C9F")
        axis.axvline(selected_threshold, color="#B23A32", linestyle="--", label="Selecionado")
        axis.set(xlabel="Limiar", ylabel=label, title=label)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Regressão Logística — avaliação em valid")
    figure.tight_layout()
    destination = report.with_suffix(".png")
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination
