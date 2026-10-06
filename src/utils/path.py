"""Caminhos centralizados do projeto e do dataset."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
EVALUATION_DIR: Path = PROJECT_ROOT / "docs" / "results"


def model_path(name: str) -> Path:
    """Retorna o caminho do arquivo pickle de um modelo.

    Args:
        name: Nome do modelo, sem diretórios ou extensão.

    Returns:
        Caminho absoluto do arquivo ``models/<name>.pkl``.
    """
    if not name or name in {".", ".."} or any(char in name for char in '/\\:*?"<>|'):
        raise ValueError("O nome do modelo deve ser um nome de arquivo válido, sem diretórios.")
    return MODELS_DIR / f"{name}.pkl"


def split_dir(split: str) -> Path:
    """Retorna o diretorio de uma divisao do dataset.

    Args:
        split: Nome da divisao, como ``train``, ``valid`` ou ``test``.

    Returns:
        Caminho absoluto do diretorio da divisao solicitada.
    """
    return DATA_DIR / split


def annotations_path(split: str) -> Path:
    """Retorna o caminho do arquivo COCO de uma divisao.

    Args:
        split: Nome da divisao do dataset.

    Returns:
        Caminho absoluto do arquivo ``_annotations.coco.json``.
    """
    return split_dir(split) / "_annotations.coco.json"
