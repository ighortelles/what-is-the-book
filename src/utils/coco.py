from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.utils.path import annotations_path, split_dir


@dataclass(frozen=True)
class Annotation:
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageRecord:
    id: int
    path: Path
    width: int
    height: int
    annotations: tuple[Annotation, ...]


def load_coco_split(split: str, category_name: str = "Book") -> list[ImageRecord]:
    """Le imagens e caixas COCO de uma divisao, filtrando uma categoria.

    Args:
        split: Nome da divisao do dataset.
        category_name: Nome da categoria COCO a ser carregada.

    Returns:
        Registros de imagens com as anotacoes da categoria solicitada.
    """
    folder = split_dir(split)
    with annotations_path(split).open(encoding="utf-8") as file:
        coco = json.load(file)
    category_ids = {item["id"] for item in coco["categories"] if item["name"] == category_name}
    if not category_ids:
        raise ValueError(f"Classe {category_name!r} nao encontrada em {folder}")
    grouped: dict[int, list[Annotation]] = {}
    for item in coco["annotations"]:
        if item["category_id"] in category_ids:
            grouped.setdefault(item["image_id"], []).append(
                Annotation(item["image_id"], item["category_id"], tuple(item["bbox"]))
            )
    return [
        ImageRecord(
            item["id"],
            folder / item["file_name"],
            item["width"],
            item["height"],
            tuple(grouped.get(item["id"], [])),
        )
        for item in coco["images"]
    ]
