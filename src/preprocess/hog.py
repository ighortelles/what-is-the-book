from __future__ import annotations

import cv2
import numpy as np

WINDOW_SIZE = (96, 128)  # largura, altura
_HOG = cv2.HOGDescriptor(WINDOW_SIZE, (16, 16), (8, 8), (8, 8), 9)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converte uma imagem colorida para escala de cinza.

    Args:
        image: Imagem BGR ou imagem ja em escala de cinza.

    Returns:
        Imagem com um unico canal de intensidade.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Aplica equalizacao adaptativa de histograma a uma imagem em cinza.

    Args:
        image: Imagem em escala de cinza.

    Returns:
        Imagem com contraste local realcado.
    """
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def resize_for_hog(image: np.ndarray) -> np.ndarray:
    """Redimensiona uma imagem para a janela fixa usada pelo HOG.

    Args:
        image: Imagem de um canal a ser redimensionada.

    Returns:
        Imagem com dimensoes definidas em ``WINDOW_SIZE``.
    """
    return cv2.resize(image, WINDOW_SIZE, interpolation=cv2.INTER_AREA)


def extract_hog(image: np.ndarray) -> np.ndarray:
    """Extrai o vetor de caracteristicas HOG de uma imagem normalizada.

    Args:
        image: Imagem em cinza e no tamanho ``WINDOW_SIZE``.

    Returns:
        Vetor unidimensional de descritores HOG.
    """
    return _HOG.compute(image).reshape(-1)


def to_hog(image: np.ndarray) -> np.ndarray:
    """Executa o pre-processamento completo e extrai caracteristicas HOG.

    Args:
        image: Recorte BGR ou em escala de cinza.

    Returns:
        Vetor unidimensional de descritores HOG.
    """
    return extract_hog(resize_for_hog(apply_clahe(to_grayscale(image))))
