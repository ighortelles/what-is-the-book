"""Tabelas e figuras didáticas para o notebook de apresentação."""

import json
from collections.abc import Sequence
from html import escape
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.detection.hog_detector import Detection
from src.evaluation.thresholds import ThresholdResult
from src.preprocess.hog import apply_clahe, resize_for_hog, to_grayscale
from src.utils.coco import ImageRecord
from src.utils.geometry import clip_box
from src.utils.path import EVALUATION_DIR


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> HTML:
    """Cria uma tabela legível com escape de textos e números formatados.

    Args:
        headers: Nomes das colunas.
        rows: Linhas já formatadas para exibição.

    Returns:
        Tabela HTML para display no notebook.
    """
    cells = "".join(f"<th>{escape(label)}</th>" for label in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return HTML(
        '<div style="overflow-x:auto"><table style="font-size:15px;text-align:left">'
        f"<thead><tr>{cells}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def dataset_gallery(records: list[ImageRecord], count: int = 15, seed: int = 42) -> Figure:
    """Mostra exemplos anotados escolhidos sem reposição no conjunto de treino.

    Args:
        records: Registros de train, para não antecipar a avaliação em test.
        count: Quantidade máxima de imagens, preferencialmente dez ou quinze.
        seed: Semente usada para escolher exemplos reproduzíveis.

    Returns:
        Figura com imagens RGB e caixas reais em verde.
    """
    if not records or count < 1:
        raise ValueError("Informe registros e uma quantidade positiva.")
    indices = np.random.default_rng(seed).choice(
        len(records), min(count, len(records)), replace=False
    )
    rows = int(np.ceil(len(indices) / 5))
    figure, axes = plt.subplots(rows, 5, figsize=(18, 3.5 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, indices):
        record = records[int(index)]
        image = cv2.imread(str(record.path))
        if image is None:
            raise ValueError(f"Imagem inválida: {record.path}")
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for annotation in record.annotations:
            x, y, width, height = clip_box(annotation.bbox, image.shape[1], image.shape[0])
            axis.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#00d36f", lw=2))
        axis.set_title(f"ID {record.id} · {len(record.annotations)} anotação(ões)", fontsize=11)
    figure.suptitle(f"{len(indices)} exemplos de train · verde = anotação real", fontsize=17)
    figure.tight_layout()
    return figure


def preprocessing_figure(record: ImageRecord) -> Figure:
    """Ilustra as transformações reais de um recorte positivo do pipeline.

    Args:
        record: Imagem de treino com ao menos uma caixa de livro.

    Returns:
        Figura com recorte BGR convertido a RGB, cinza, CLAHE e tamanho HOG.
    """
    from src.detection.hog_detector import _crop
    from src.preprocess.image import normalize_size

    image = cv2.imread(str(record.path))
    if image is None or not record.annotations:
        raise ValueError("Escolha uma imagem válida com anotação de livro.")
    normalized, (scale_x, scale_y) = normalize_size(image)
    x, y, width, height = clip_box(record.annotations[0].bbox, image.shape[1], image.shape[0])
    crop = _crop(normalized, (x * scale_x, y * scale_y, width * scale_x, height * scale_y))
    gray = to_grayscale(crop)
    contrast = apply_clahe(gray)
    resized = resize_for_hog(contrast)
    images = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), gray, contrast, resized]
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    titles = ["Recorte positivo", "Escala de cinza", "Contraste local (CLAHE)", "96 × 128 → HOG"]
    for axis, processed, title in zip(axes, images, titles, strict=True):
        axis.imshow(processed, cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    return figure


def threshold_figure(
    reports: dict[str, list[ThresholdResult]], selections: dict[str, ThresholdResult | None]
) -> Figure:
    """Mostra curvas com eixos separados para margens e probabilidades.

    Args:
        reports: Resultados de validação por experimento.
        selections: Resultados escolhidos pela restrição de recall, se existentes.

    Returns:
        Figura com precisão, recall e FP para cada experimento.
    """
    figure, axes = plt.subplots(len(reports), 3, figsize=(15, 3.2 * len(reports)), squeeze=False)
    for row, (name, results) in enumerate(reports.items()):
        xs = [item.threshold for item in results]
        series = [
            [100 * item.metrics.precision for item in results],
            [100 * item.metrics.recall for item in results],
            [item.metrics.false_positives for item in results],
        ]
        for axis, values, label in zip(
            axes[row], series, ["Precisão (%)", "Recall (%)", "Falsos positivos"], strict=True
        ):
            axis.plot(xs, values, "o-", color="#245c9f", ms=4)
            if selections.get(name) is not None:
                axis.axvline(selections[name].threshold, color="#c3473a", ls="--")
            if label == "Recall (%)":
                axis.axhline(50, color="#666666", ls=":", label="Recall mínimo")
            axis.set(xlabel="Limiar na escala do modelo", ylabel=label, title=f"{name}\n{label}")
            axis.grid(alpha=0.2)
    figure.suptitle("Validação · linha vermelha = limiar selecionado", fontsize=16)
    figure.tight_layout()
    return figure


def comparison_figure(selections: dict[str, ThresholdResult | None]) -> Figure:
    """Compara somente configurações elegíveis sob o mesmo mínimo de recall.

    Args:
        selections: Resultados escolhidos em valid por experimento.

    Returns:
        Figura com precisão, recall e F1 em painéis separados.
    """
    eligible = {name: result for name, result in selections.items() if result is not None}
    if not eligible:
        raise ValueError("Nenhum experimento atende ao recall mínimo.")
    names = [name.replace(" — ", "\n") for name in eligible]
    metrics = [result.metrics for result in eligible.values()]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    positions = np.arange(len(names))
    colors = ["#245c9f", "#20a17b", "#d97b37", "#8b5da9"]
    for index, (field, label) in enumerate(
        [("precision", "Precisão (%)"), ("recall", "Recall (%)"), ("f1_score", "F1 (%)")]
    ):
        values = [100 * getattr(item, field) for item in metrics]
        bars = axes[index].bar(
            positions, values, color=[colors[i % len(colors)] for i in range(len(names))]
        )
        axes[index].bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3)
        axes[index].set_xticks(positions, names, fontsize=9)
        axes[index].set(ylabel=label, title=label, ylim=(0, max(values) * 1.2 + 1))
        axes[index].grid(axis="y", alpha=0.2)
    figure.suptitle("Modelos elegíveis · todos com recall ≥ 50% em valid", fontsize=16)
    figure.tight_layout()
    return figure


def detection_figure(record: ImageRecord, predictions: list[Detection], title: str) -> Figure:
    """Separa anotações reais das previsões, sem esconder falsos alarmes.

    Args:
        record: Registro de validação com caixas reais.
        predictions: Todas as caixas produzidas no limiar selecionado.
        title: Nome do modelo ou descrição da configuração.

    Returns:
        Figura lado a lado: verdade de referência e todas as previsões.
    """
    image = cv2.imread(str(record.path))
    if image is None:
        raise ValueError(f"Imagem inválida: {record.path}")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis in axes:
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.axis("off")
    for annotation in record.annotations:
        x, y, width, height = annotation.bbox
        axes[0].add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#00b769", lw=2))
    for prediction in predictions:
        x, y, width, height = prediction.bbox
        axes[1].add_patch(Rectangle((x, y), width, height, fill=False, edgecolor="#e74646", lw=1.5))
    axes[0].set_title(f"Anotações reais: {len(record.annotations)}")
    axes[1].set_title(f"Todas as previsões: {len(predictions)}")
    figure.suptitle(f"{title} · imagem de valid ID {record.id}")
    figure.tight_layout()
    return figure


def validation_test_figure(
    validation: dict[str, ThresholdResult], test: dict[str, ThresholdResult]
) -> Figure:
    """Compara valid e test nos mesmos limiares, sem selecionar modelos pelo teste.

    Args:
        validation: Métricas de valid nos limiares escolhidos previamente.
        test: Métricas finais de test para os mesmos experimentos e limiares.

    Returns:
        Figura com precisão, recall e F1 das duas divisões em painéis separados.
    """
    if not test or test.keys() != validation.keys():
        raise ValueError("Informe os mesmos experimentos nas duas divisões.")
    if any(validation[name].threshold != result.threshold for name, result in test.items()):
        raise ValueError("Os limiares devem permanecer iguais entre valid e test.")
    names = list(test)
    positions = np.arange(len(names))
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, field, label in zip(
        axes,
        ["precision", "recall", "f1_score"],
        ["Precisão (%)", "Recall (%)", "F1 (%)"],
        strict=True,
    ):
        for offset, results, color, split in [
            (-0.18, validation, "#245c9f", "valid — seleção"),
            (0.18, test, "#20a17b", "test — avaliação final"),
        ]:
            values = [100 * getattr(results[name].metrics, field) for name in names]
            bars = axis.bar(positions + offset, values, width=0.36, color=color, label=split)
            axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=9)
        maximum = max(
            100 * getattr(results[name].metrics, field)
            for results in [validation, test]
            for name in names
        )
        axis.set_xticks(positions, [name.replace(" — ", "\n") for name in names], fontsize=9)
        axis.set(ylabel=label, title=label, ylim=(0, maximum * 1.2 + 1))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=9)
    figure.suptitle("Valid × test · mesmos pesos, janelas e limiares", fontsize=16)
    figure.tight_layout()
    return figure


def latest_manifest() -> dict:
    """Carrega o manifesto mais recente sem escolher versões por suposição.

    Args:
        Nenhum.

    Returns:
        Protocolo e caminhos dos resultados efetivamente executados.
    """
    files = sorted(EVALUATION_DIR.glob("precision_run_*.json"))
    if not files:
        raise FileNotFoundError("Execute python -m src.evaluation.run_precision primeiro.")
    with files[-1].open(encoding="utf-8") as file:
        return json.load(file)


def export_figure(figure: Figure, name: str) -> Path:
    """Exporta um gráfico para o artigo sem substituir figuras anteriores.

    Args:
        figure: Figura Matplotlib já criada.
        name: Nome base do PNG, sem diretórios nem extensão.

    Returns:
        Caminho da figura exportada com sufixo numérico quando necessário.
    """
    if not name or any(char in name for char in '/\\:*?"<>|'):
        raise ValueError("Informe um nome de figura válido.")
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    version = 0
    while True:
        suffix = f"_{version:03d}" if version else ""
        path = EVALUATION_DIR / f"{name}{suffix}.png"
        try:
            file = path.open("xb")
        except FileExistsError:
            version += 1
            continue
        with file:
            figure.savefig(file, format="png", dpi=150, bbox_inches="tight")
        return path
