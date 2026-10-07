"""Protocolo final congelado, rastreabilidade e reutilização da avaliação em test."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from sklearn.dummy import DummyClassifier

from src.evaluation.detection import calculate_metrics
from src.evaluation.test_set import evaluate_test_models, save_test_report
from src.evaluation.thresholds import ThresholdResult
from src.utils.coco import Annotation, ImageRecord
from src.utils.model_io import save_model
from src.utils.presentation import validation_test_figure


class TestSetTests(unittest.TestCase):
    def test_frozen_thresholds_cache_and_recommendation_from_validation(self) -> None:
        """Mantém os limiares e a recomendação de valid e não repete inferência idêntica.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        model = DummyClassifier(strategy="constant", constant=1).fit(np.zeros((2, 1)), [0, 1])
        manifest = {
            "split": "valid",
            "iou_threshold": 0.5,
            "recommended_model": "unit_lr",
            "experiments": [
                {"model": "unit_lr", "label": "LR", "selected": {"threshold": 6.0}},
                {"model": "unit_mlp", "label": "MLP", "selected": {"threshold": 0.495}},
            ],
        }
        metrics = {
            "unit_lr": [ThresholdResult(6.0, calculate_metrics(1, 1, 0))],
            "unit_mlp": [ThresholdResult(0.495, calculate_metrics(1, 0, 0))],
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image_path = directory / "image.png"
            cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
            record = ImageRecord(1, image_path, 32, 32, (Annotation(1, 1, (0, 0, 16, 16)),))
            with (
                patch("src.utils.path.MODELS_DIR", directory),
                patch("src.evaluation.test_set.EVALUATION_DIR", directory),
                patch(
                    "src.evaluation.test_set.load_coco_split", return_value=[record]
                ) as load_split,
                patch(
                    "src.evaluation.test_set.evaluate_models_thresholds", return_value=metrics
                ) as evaluate,
            ):
                save_model(model, "unit_lr")
                save_model(model, "unit_mlp")
                first_path, first = evaluate_test_models(manifest)
                saved_bytes = first_path.read_bytes()
                cached_path, cached = evaluate_test_models(manifest)
                self.assertEqual(first_path, cached_path)
                self.assertEqual(first, cached)
                self.assertEqual(evaluate.call_count, 1)
                self.assertTrue(all(call.args == ("test",) for call in load_split.call_args_list))
                self.assertEqual(
                    evaluate.call_args.args[2], {"unit_lr": [6.0], "unit_mlp": [0.495]}
                )
                self.assertEqual(first["protocol"]["recommended_model_from_valid"], "unit_lr")
                self.assertEqual(first["protocol"]["threshold_source"], "valid")
                self.assertEqual(len(first["protocol"]["models"][0]["model_sha256"]), 64)

                # Mudar as imagens invalida o cache, preservando o relatório anterior.
                cv2.imwrite(str(image_path), np.full((32, 32, 3), 255, dtype=np.uint8))
                changed_path, changed = evaluate_test_models(manifest)
                self.assertNotEqual(first_path, changed_path)
                self.assertNotEqual(
                    first["protocol"]["dataset_sha256"], changed["protocol"]["dataset_sha256"]
                )
                self.assertEqual(evaluate.call_count, 2)
                self.assertEqual(first_path.read_bytes(), saved_bytes)

    def test_test_cannot_be_the_source_of_selection(self) -> None:
        """Recusa um manifesto que declara test como origem dos ajustes.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        with self.assertRaisesRegex(ValueError, "selecionados em valid"):
            evaluate_test_models({"split": "test"})

    def test_missing_validation_threshold_is_not_silently_replaced(self) -> None:
        """Não adota um fallback para comparar um modelo sem limiar congelado.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        manifest = {
            "split": "valid",
            "experiments": [{"label": "LR", "selected": None}],
        }
        with patch("src.evaluation.test_set.load_coco_split", return_value=[]):
            with self.assertRaisesRegex(ValueError, "Defina um limiar em valid"):
                evaluate_test_models(manifest)

    def test_report_is_versioned_and_never_overwritten(self) -> None:
        """Verifica salvamentos sucessivos e conteúdo preservado.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        with tempfile.TemporaryDirectory() as temporary:
            with patch("src.evaluation.test_set.EVALUATION_DIR", Path(temporary)):
                first = save_test_report({"protocol": {"split": "test"}, "results": []})
                content = first.read_bytes()
                second = save_test_report({"protocol": {"split": "test"}, "results": []})
                self.assertNotEqual(first, second)
                self.assertEqual(first.read_bytes(), content)
                self.assertEqual(json.loads(content)["protocol"]["split"], "test")

    def test_comparison_plot_rejects_different_thresholds(self) -> None:
        """Não apresenta como comparação congelada resultados com limiares diferentes.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        validation = {"LR": ThresholdResult(6.0, calculate_metrics(1, 1, 0))}
        test = {"LR": ThresholdResult(10.0, calculate_metrics(1, 0, 0))}
        with self.assertRaisesRegex(ValueError, "limiares devem permanecer iguais"):
            validation_test_figure(validation, test)


if __name__ == "__main__":
    unittest.main()
