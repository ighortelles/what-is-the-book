"""Regressões de amostragem, cobertura e coordenadas do detector."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from sklearn.dummy import DummyClassifier

from src.detection.classifiers import build_classifier
from src.detection.hard_negatives import mine_hard_negatives
from src.detection.hog_detector import (
    Detection,
    _crop,
    detect,
    detect_many,
    non_maximum_suppression,
    training_features,
)
from src.detection.windows import DetectorConfig, axis_positions, window_boxes
from src.evaluation.detection import calculate_metrics, evaluate_detector
from src.evaluation.thresholds import (
    ThresholdResult,
    evaluate_models_thresholds,
    evaluate_thresholds,
    load_threshold_report,
    save_threshold_report,
    select_threshold,
)
from src.preprocess.image import normalize_size
from src.utils.coco import Annotation, ImageRecord
from src.utils.geometry import overlap_matrices
from src.utils.model_io import load_model, save_model


def mean_feature(image: np.ndarray) -> np.ndarray:
    """Representa a intensidade de um recorte para testar sua ocupação.

    Args:
        image: Recorte BGR.

    Returns:
        Vetor de um valor com a intensidade média.
    """
    return np.asarray([image.mean()], dtype=np.float32)


class PipelineTests(unittest.TestCase):
    def test_regularization_is_explicit_and_validated(self) -> None:
        """Confirma que C menor é repassado ao modelo sem modificar o padrão.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        self.assertEqual(build_classifier("logistic_regression")["model"].C, 1.0)
        self.assertEqual(build_classifier("logistic_regression", 0.01)["model"].C, 0.01)
        for value in [0.0, -1.0, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                build_classifier("logistic_regression", value)

    def test_hard_negatives_exclude_annotated_book_and_respect_budget(self) -> None:
        """Exclui fragmentos de livros e limita negativos difíceis por imagem.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[:, :64] = 255
        model = DummyClassifier().fit(np.zeros((2, 1)), [0, 1])
        model.detector_config_ = DetectorConfig(max_side=128)
        predictions = [
            Detection((0, 0, 32, 128), 99.0),
            Detection((96, 0, 32, 128), 5.0),
            Detection((64, 0, 32, 128), 4.0),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.png"
            cv2.imwrite(str(path), image)
            record = ImageRecord(7, path, 128, 128, (Annotation(7, 1, (0, 0, 64, 128)),))
            with (
                patch("src.detection.hard_negatives.detect", return_value=predictions),
                patch("src.detection.hard_negatives.to_hog", side_effect=mean_feature),
            ):
                result = mine_hard_negatives([record], model, per_image=1)
        self.assertEqual(result.features.shape, (1, 1))
        self.assertEqual(result.features[0, 0], 0.0)
        self.assertEqual(result.image_ids, (7,))
        self.assertEqual(result.source_boxes, ((7, (96, 0, 32, 128)),))

    def test_shared_hog_matches_separate_models(self) -> None:
        """Confirma equivalência entre detecção compartilhada e individual.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        models = {
            str(label): DummyClassifier(strategy="constant", constant=label).fit(
                np.zeros((2, 5940)), [0, 1]
            )
            for label in [0, 1]
        }
        config = DetectorConfig(max_side=128, window_sizes=((32, 64),), size_factors=(1.0,))
        for model in models.values():
            model.detector_config_ = config
        predictions = detect_many(image, models, {name: -0.5 for name in models})
        for name, model in models.items():
            self.assertEqual(predictions[name], detect(image, model, -0.5))
        models["0"].detector_config_ = DetectorConfig(max_side=96)
        with self.assertRaises(ValueError):
            detect_many(image, models, {name: -0.5 for name in models})

    def test_shared_validation_matches_separate_evaluations(self) -> None:
        """Confirma correspondência de métricas para múltiplos modelos e limiares.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        models = {
            str(label): DummyClassifier(strategy="constant", constant=label).fit(
                np.zeros((2, 5940)), [0, 1]
            )
            for label in [0, 1]
        }
        for model in models.values():
            model.detector_config_ = DetectorConfig(
                max_side=128, window_sizes=((128, 64),), size_factors=(1.0,)
            )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.png"
            cv2.imwrite(str(path), image)
            records = [ImageRecord(1, path, 128, 64, (Annotation(1, 1, (0, 0, 128, 64)),))]
            results = evaluate_models_thresholds(
                records, models, {name: [-0.5, 0.0, 0.5] for name in models}
            )
            for name, model in models.items():
                for item in results[name]:
                    self.assertEqual(
                        item.metrics, evaluate_detector(records, model, item.threshold)
                    )

    def test_contained_window_is_not_background(self) -> None:
        """Confirma que IoU baixa não torna um fragmento de livro negativo.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        overlaps, occupancy = overlap_matrices(
            np.asarray([[10, 10, 20, 20]], dtype=float),
            np.asarray([[0, 0, 500, 500]], dtype=float),
        )
        self.assertLess(overlaps[0, 0], 0.1)
        self.assertEqual(occupancy[0, 0], 1.0)

    def test_training_negatives_do_not_contain_book_pixels(self) -> None:
        """Verifica recortes amostrados com livro e fundo conhecidos.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[:, :96] = 255
        config = DetectorConfig(max_side=128, window_sizes=((16, 32),), size_factors=(1.0,))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.png"
            cv2.imwrite(str(path), image)
            record = ImageRecord(1, path, 128, 128, (Annotation(1, 1, (0, 0, 96, 128)),))
            with patch("src.detection.hog_detector.to_hog", side_effect=mean_feature):
                features, labels = training_features([record], config=config)
        self.assertTrue((features[labels == 0] <= 2.55).all())
        self.assertGreater(int((labels == 0).sum()), 0)
        self.assertLessEqual(int((labels == 0).sum()), 15)
        self.assertGreater(int((labels == 1).sum()), 0)

    def test_normalization_and_coordinate_restoration(self) -> None:
        """Confirma o retorno das caixas às dimensões da imagem original.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        image = np.zeros((320, 640, 3), dtype=np.uint8)
        config = DetectorConfig(max_side=128, window_sizes=((128, 64),), size_factors=(1.0,))
        normalized, scales = normalize_size(image, 128)
        self.assertEqual(normalized.shape[:2], (64, 128))
        self.assertEqual(scales, (0.2, 0.2))
        model = DummyClassifier(strategy="constant", constant=1).fit(np.zeros((2, 5940)), [0, 1])
        model.detector_config_ = config
        predictions = detect(image, model, batch_size=1)
        self.assertEqual([item.bbox for item in predictions], [(0, 0, 640, 320)])

    def test_batch_size_does_not_change_detections(self) -> None:
        """Confirma que previsões em lotes preservam as caixas retornadas.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        image = np.zeros((163, 217, 3), dtype=np.uint8)
        model = DummyClassifier(strategy="constant", constant=1).fit(np.zeros((2, 5940)), [0, 1])
        model.detector_config_ = DetectorConfig(max_side=96, window_sizes=((32, 48),))
        single = detect(image, model, batch_size=1)
        batch = detect(image, model, batch_size=32)
        self.assertEqual(single, batch)
        self.assertGreater(len(batch), 1)
        for item in batch:
            x, y, width, height = item.bbox
            self.assertLessEqual(x + width, 217)
            self.assertLessEqual(y + height, 163)

    def test_border_windows_and_crop(self) -> None:
        """Verifica busca na borda e recortes parcialmente fora da imagem.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        self.assertEqual(axis_positions(100, 30, 32), [0, 32, 64, 70])
        self.assertEqual(_crop(np.zeros((10, 10, 3)), (-2, -2, 3, 3)).shape[:2], (1, 1))
        boxes = list(window_boxes(100, 100, DetectorConfig(), 32))
        self.assertIn((0, 0, 100, 100), boxes)

    def test_nms_keeps_best_duplicate_and_distinct_object(self) -> None:
        """Confirma a remoção de duplicatas sem eliminar uma caixa distante.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        best = Detection((0, 0, 100, 100), 2.0)
        duplicate = Detection((1, 1, 100, 100), 1.0)
        other = Detection((200, 200, 100, 100), 1.5)
        self.assertEqual(non_maximum_suppression([duplicate, other, best]), [best, other])

    def test_saved_model_retains_detector_settings(self) -> None:
        """Verifica persistência dos tamanhos de busca junto com o modelo.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        model = DummyClassifier(strategy="constant", constant=1).fit(np.zeros((2, 5940)), [0, 1])
        model.detector_config_ = DetectorConfig(window_sizes=((200, 300),))
        with tempfile.TemporaryDirectory() as temporary:
            with patch("src.utils.path.MODELS_DIR", Path(temporary)):
                destination = save_model(model, "unit_model")
                restored = load_model(destination.stem)
                self.assertEqual(restored.detector_config_, model.detector_config_)

    def test_threshold_comparison_matches_independent_evaluations(self) -> None:
        """Confirma que reutilizar NMS preserva a avaliação em cada limiar.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        model = DummyClassifier(strategy="constant", constant=1).fit(np.zeros((2, 5940)), [0, 1])
        model.detector_config_ = DetectorConfig(
            max_side=128, window_sizes=((128, 64),), size_factors=(1.0,)
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.png"
            cv2.imwrite(str(path), image)
            record = ImageRecord(1, path, 128, 64, (Annotation(1, 1, (0, 0, 128, 64)),))
            results = evaluate_thresholds([record], model, [0.0, 0.5, 0.6])
            for item in results:
                expected = evaluate_detector([record], model, item.threshold)
                self.assertEqual(item.metrics, expected)
            self.assertEqual(results[-1].metrics.false_negatives, 1)

    def test_selection_respects_recall_floor(self) -> None:
        """Evita escolher precisão alta que descarta quase todos os livros.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        baseline = ThresholdResult(0.0, calculate_metrics(6, 10, 4))
        acceptable = ThresholdResult(2.0, calculate_metrics(5, 1, 5))
        too_strict = ThresholdResult(5.0, calculate_metrics(1, 0, 9))
        self.assertEqual(select_threshold([baseline, acceptable, too_strict]), acceptable)
        with self.assertRaises(ValueError):
            select_threshold([too_strict])

    def test_threshold_report_preserves_previous_file(self) -> None:
        """Confirma que salvar tabelas repetidas cria relatórios distintos.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        results = [ThresholdResult(0.0, calculate_metrics(3, 1, 2))]
        with tempfile.TemporaryDirectory() as temporary:
            with patch("src.evaluation.thresholds.EVALUATION_DIR", Path(temporary)):
                first = save_threshold_report(results, "logistic_regression_001")
                content = first.read_bytes()
                second = save_threshold_report(results, "logistic_regression_001")
                self.assertNotEqual(first, second)
                self.assertEqual(first.read_bytes(), content)
                self.assertEqual(load_threshold_report(first), results)


if __name__ == "__main__":
    unittest.main()
