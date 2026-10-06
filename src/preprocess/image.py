"""Normalização espacial das imagens, preservando suas proporções."""

import cv2
import numpy as np


def normalize_size(
    image: np.ndarray, max_side: int = 640
) -> tuple[np.ndarray, tuple[float, float]]:
    """Normaliza o maior lado da imagem e informa as escalas aplicadas.

    Args:
        image: Imagem BGR de entrada.
        max_side: Tamanho do maior lado após o redimensionamento.

    Returns:
        Imagem redimensionada e fatores de escala de largura e altura.
    """
    if image.size == 0 or max_side < 16:
        raise ValueError("A imagem deve ser válida e max_side deve ser pelo menos 16.")
    height, width = image.shape[:2]
    factor = max_side / max(height, width)
    new_width = max(1, round(width * factor))
    new_height = max(1, round(height * factor))
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    return resized, (new_width / width, new_height / height)
