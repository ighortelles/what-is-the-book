"""Configuração e geração de janelas adequadas às caixas do treinamento."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from src.utils.coco import ImageRecord
from src.utils.geometry import clip_box


@dataclass(frozen=True)
class DetectorConfig:
    """Parâmetros de busca, salvos junto com cada classificador novo."""

    max_side: int = 640
    window_sizes: tuple[tuple[int, int], ...] = (
        (48, 64),
        (64, 96),
        (96, 128),
        (128, 192),
        (192, 256),
        (256, 384),
        (384, 512),
        (512, 512),
        (128, 96),
        (256, 192),
        (384, 256),
    )
    size_factors: tuple[float, ...] = (0.8, 1.0, 1.25)


def fit_window_config(
    records: list[ImageRecord], n_templates: int = 12, max_side: int = 640
) -> DetectorConfig:
    """Estima tamanhos de janelas usando apenas as anotações de treinamento.

    Args:
        records: Registros da divisão train.
        n_templates: Quantidade máxima de grupos de tamanhos de caixas.
        max_side: Maior lado das imagens normalizadas.

    Returns:
        Configuração com centros de grupos de larguras e alturas em escala log.
    """
    if n_templates < 1 or max_side < 16:
        raise ValueError("n_templates deve ser positivo e max_side deve ser pelo menos 16.")
    sizes = []
    for record in records:
        scale = max_side / max(record.width, record.height)
        for annotation in record.annotations:
            _, _, width, height = clip_box(annotation.bbox, record.width, record.height)
            if width > 0 and height > 0:
                sizes.append((max(16, width * scale), max(16, height * scale)))
    if not sizes:
        raise ValueError("Não há caixas válidas para estimar os tamanhos das janelas.")
    unique = np.unique(np.log(np.asarray(sizes)), axis=0)
    clusters = KMeans(n_clusters=min(n_templates, len(unique)), random_state=42, n_init=10)
    clusters.fit(np.log(np.asarray(sizes)))
    templates = sorted(
        {tuple(max(16, round(v)) for v in row) for row in np.exp(clusters.cluster_centers_)}
    )
    return DetectorConfig(max_side=max_side, window_sizes=tuple(templates))


def axis_positions(length: int, window: int, step: int) -> list[int]:
    """Gera posições de busca incluindo a borda final da imagem.

    Args:
        length: Comprimento do eixo da imagem.
        window: Comprimento da janela nesse eixo.
        step: Deslocamento entre posições consecutivas.

    Returns:
        Posições válidas, incluindo a última janela junto à borda.
    """
    if window < 1 or length < 1 or step < 1:
        raise ValueError("O eixo, a janela e o passo devem ser positivos.")
    if window > length:
        return []
    positions = list(range(0, length - window + 1, step))
    if positions[-1] != length - window:
        positions.append(length - window)
    return positions


def window_boxes(
    width: int, height: int, config: DetectorConfig, step: int = 32
) -> Iterator[tuple[int, int, int, int]]:
    """Gera janelas verticais, horizontais e quadradas na imagem normalizada.

    Args:
        width: Largura da imagem normalizada.
        height: Altura da imagem normalizada.
        config: Tamanhos de janelas e fatores de escala.
        step: Passo máximo de busca; janelas pequenas usam um passo menor.

    Returns:
        Iterador de caixas x, y, largura, altura dentro da imagem.
    """
    if step < 1:
        raise ValueError("O passo da busca deve ser positivo.")
    sizes = {(width, height)}
    for w, h in config.window_sizes:
        for factor in config.size_factors:
            sizes.add((max(16, round(w * factor)), max(16, round(h * factor))))
    for w, h in sorted(sizes):
        if w > width or h > height:
            continue
        stride = min(step, max(16, min(w, h) // 4))
        for y in axis_positions(height, h, stride):
            for x in axis_positions(width, w, stride):
                yield x, y, w, h
