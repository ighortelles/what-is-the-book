"""Rótulos descritivos e comparação independente dos modelos no Streamlit."""

import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from sklearn.dummy import DummyClassifier
from streamlit.testing.v1 import AppTest

from src.app.interface import cached_model, model_family, model_label, run_comparison
from src.detection.hog_detector import detect, detect_many
from src.detection.windows import DetectorConfig
from src.utils.model_io import save_model
from src.utils.path import PROJECT_ROOT


def constant_model(label: int, max_side: int = 128) -> DummyClassifier:
    """Cria um classificador mínimo para testar detecção e interface rapidamente.

    Args:
        label: Classe constante, zero para fundo ou um para livro.
        max_side: Escala usada para separar grupos de configurações.

    Returns:
        Classificador treinado com configuração e limiar salvos.
    """
    model = DummyClassifier(strategy="constant", constant=label).fit(np.zeros((2, 5940)), [0, 1])
    model.detector_config_ = DetectorConfig(
        max_side=max_side, window_sizes=((max_side, max_side // 2),), size_factors=(1.0,)
    )
    model.score_threshold_ = 0.0
    return model


class AppTests(unittest.TestCase):
    def test_labels_describe_experiments_instead_of_version_numbers(self) -> None:
        """Diferencia snapshots conhecidos sem usar números como nome principal.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        self.assertIn("baseline inicial", model_label("logistic_regression"))
        self.assertIn("15 negativos", model_label("logistic_regression_001"))
        self.assertIn("30 negativos", model_label("logistic_regression_002"))
        self.assertIn("mesmos pesos", model_label("logistic_regression_003"))
        self.assertIn("C=0,01", model_label("logistic_regression_004"))
        self.assertIn("172 negativos difíceis", model_label("logistic_regression_005"))
        self.assertIn("128/64", model_label("mlp"))
        self.assertEqual(model_family("mlp_001"), "mlp")
        model = SimpleNamespace(
            training_metadata_={"negatives_per_image": 60, "hard_negatives": 10},
            named_steps={"model": SimpleNamespace(C=0.1)},
        )
        label = model_label("logistic_regression_006", model)
        self.assertIn("60 negativos", label)
        self.assertIn("10 negativos difíceis", label)
        self.assertIn("C=0.1", label)

    def test_comparison_uses_individual_thresholds_and_shared_hog(self) -> None:
        """Verifica os limiares de cada modelo e uma única extração por grupo.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        models = {"logistic_regression_004": constant_model(1), "mlp": constant_model(1)}
        thresholds = {"logistic_regression_004": 0.0, "mlp": 0.6}
        with patch("src.app.interface.detect_many", wraps=detect_many) as shared:
            result = run_comparison(image, models, thresholds)
        self.assertEqual(shared.call_count, 1)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)
        for item in result.models:
            self.assertIsNone(item.error)
            self.assertEqual(item.threshold, thresholds[item.name])
            self.assertEqual(
                item.detections, detect(image, models[item.name], thresholds[item.name])
            )
        self.assertTrue(result.models[0].detections)
        self.assertFalse(result.models[1].detections)

    def test_different_configs_and_loading_failure_do_not_block_other_models(self) -> None:
        """Mantém os grupos separados e não confunde erro com ausência de livros.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        models = {"logistic_regression_004": constant_model(1), "mlp": constant_model(0, 96)}
        with patch("src.app.interface.detect_many", wraps=detect_many) as shared:
            result = run_comparison(
                image, models, {name: 0.0 for name in models}, load_errors={"broken": "ImportError"}
            )
        self.assertEqual(shared.call_count, 2)
        self.assertEqual(len(result.models), 3)
        broken = next(item for item in result.models if item.name == "broken")
        self.assertEqual(broken.error, "ImportError")
        self.assertIsNone(broken.threshold)
        self.assertTrue(
            next(item for item in result.models if item.name.startswith("logistic")).detections
        )

    def test_failed_prediction_is_isolated_inside_shared_group(self) -> None:
        """Repete individualmente um grupo que falhou para preservar os modelos válidos.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        good = constant_model(1)
        broken = SimpleNamespace(
            detector_config_=good.detector_config_, classes_=np.asarray([0, 1])
        )
        result = run_comparison(
            image, {"good": good, "broken": broken}, {"good": 0.0, "broken": 0.0}
        )
        by_name = {item.name: item for item in result.models}
        self.assertIsNotNone(by_name["broken"].error)
        self.assertIsNone(by_name["good"].error)
        self.assertTrue(by_name["good"].detections)

    def test_ui_runs_all_models_and_invalidates_outdated_results(self) -> None:
        """Testa upload único, persistência no rerun e invalidação dos parâmetros.

        Args:
            Nenhum.

        Returns:
            Nenhum.
        """
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        encoded, content = cv2.imencode(".png", image)
        self.assertTrue(encoded)
        upload = BytesIO(content.tobytes())
        cached_model.clear()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with (
                patch("src.utils.path.MODELS_DIR", directory),
                patch("src.app.interface.MODELS_DIR", directory),
                patch("src.app.interface.st.file_uploader", return_value=upload),
                patch("src.app.interface.detect_many", wraps=detect_many) as shared,
            ):
                save_model(constant_model(1), "logistic_regression_004")
                save_model(constant_model(0), "mlp")
                save_model(constant_model(1), "linear_svm")
                app = AppTest.from_file(str(PROJECT_ROOT / "src/app/main.py"), default_timeout=30)
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.selectbox), 0)
                self.assertEqual(len(app.number_input), 2)
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(shared.call_count, 1)
                results = app.dataframe[1].value
                self.assertEqual(
                    set(results["Arquivo"]), {"logistic_regression_004.pkl", "mlp.pkl"}
                )
                self.assertEqual(list(results["Detecção de livro"]), ["Sim", "Não"])
                app.run()
                self.assertEqual(shared.call_count, 1)
                self.assertEqual(len(app.dataframe[1].value), 2)
                app.sidebar.number_input[0].set_value(0.7).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.dataframe), 1)
                self.assertEqual(shared.call_count, 1)
        cached_model.clear()


if __name__ == "__main__":
    unittest.main()
