from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]


def clip_box(box: Box, width: int, height: int) -> Box:
    """Limita uma caixa à região visível da imagem.

    Args:
        box: Caixa no formato x, y, largura, altura.
        width: Largura da imagem.
        height: Altura da imagem.

    Returns:
        Caixa recortada, com área zero quando não houver interseção.
    """
    x, y, w, h = box
    left, top = min(width, max(0.0, x)), min(height, max(0.0, y))
    right = min(width, max(0.0, x + max(0.0, w)))
    bottom = min(height, max(0.0, y + max(0.0, h)))
    return left, top, max(0.0, right - left), max(0.0, bottom - top)


def overlap_matrices(candidates: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula IoU e ocupação de cada janela pelas caixas anotadas.

    Args:
        candidates: Matriz N por 4 de janelas no formato x, y, largura, altura.
        targets: Matriz M por 4 de anotações no mesmo formato.

    Returns:
        Matrizes N por M com IoU e interseção dividida pela área da janela.
    """
    left = np.maximum(candidates[:, None, :2], targets[None, :, :2])
    right = np.minimum(
        candidates[:, None, :2] + candidates[:, None, 2:],
        targets[None, :, :2] + targets[None, :, 2:],
    )
    intersection = np.prod(np.maximum(0.0, right - left), axis=2)
    candidate_area = np.prod(candidates[:, 2:], axis=1)[:, None]
    target_area = np.prod(targets[:, 2:], axis=1)[None, :]
    union = candidate_area + target_area - intersection
    overlaps = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    occupancy = np.divide(
        intersection,
        candidate_area,
        out=np.zeros_like(intersection),
        where=candidate_area > 0,
    )
    return overlaps, occupancy


def iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    """Calcula Intersection over Union entre duas caixas delimitadoras.

    Args:
        first: Primeira caixa no formato ``x, y, largura, altura``.
        second: Segunda caixa no formato ``x, y, largura, altura``.

    Returns:
        Proporcao entre a area de intersecao e a area de uniao das caixas.
    """
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0
