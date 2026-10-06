"""Comparação de todos os modelos salvos sobre uma mesma imagem."""

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter

import cv2
import numpy as np
import streamlit as st

from src.detection.hog_detector import Detection, detect, detect_many
from src.detection.windows import DetectorConfig
from src.utils.model_io import load_model
from src.utils.path import MODELS_DIR, model_path

MODEL_LABELS: dict[str, str] = {
    "logistic_regression": "Regressão Logística",
    "mlp": "MLP",
}

SNAPSHOT_LABELS: dict[str, str] = {
    "logistic_regression": "Regressão — baseline inicial (672 recortes)",
    "logistic_regression_001": "Regressão — protocolo corrigido, 15 negativos",
    "logistic_regression_002": "Regressão — referência, 30 negativos (C=1)",
    "logistic_regression_003": "Regressão — referência reavaliada (mesmos pesos)",
    "logistic_regression_004": "Regressão — regularização forte (C=0,01)",
    "logistic_regression_005": "Regressão — 30 negativos + 172 negativos difíceis",
    "mlp": "MLP — HOG, camadas 128/64, 30 negativos",
}


@dataclass(frozen=True)
class ModelResult:
    """Previsões ou erro de um modelo, sem misturar decisões de outros modelos."""

    name: str
    label: str
    threshold: float | None
    detections: list[Detection]
    error: str | None = None


@dataclass(frozen=True)
class ComparisonResult:
    """Resultados individuais e tempo total da inferência compartilhada."""

    models: list[ModelResult]
    elapsed_seconds: float


def model_family(name: str) -> str:
    """Identifica o classificador a partir do nome de um arquivo versionado.

    Args:
        name: Nome do arquivo, sem extensão, com ou sem sufixo numérico.

    Returns:
        Nome do classificador reconhecido ou o próprio nome do arquivo.
    """
    base, _, version = name.rpartition("_")
    if base in MODEL_LABELS and version.isdigit() and len(version) >= 3:
        return base
    return name


def model_label(name: str, classifier: object | None = None) -> str:
    """Obtém o nome do classificador para exibição na interface.

    Args:
        name: Nome do arquivo do modelo, sem extensão.
        classifier: Pipeline opcional para descrever os metadados de novas versões.

    Returns:
        Descrição do experimento, mantendo o nome técnico separado na interface.
    """
    family = model_family(name)
    if name in SNAPSHOT_LABELS:
        return SNAPSHOT_LABELS[name]
    metadata = getattr(classifier, "training_metadata_", {})
    if not metadata:
        return f"{MODEL_LABELS.get(family, name)} — configuração não documentada"
    negatives = metadata.get("negatives_per_image", "não informado")
    hard_negatives = metadata.get("hard_negatives", 0)
    if family == "mlp":
        estimator = getattr(classifier, "named_steps", {}).get("model")
        layers = getattr(estimator, "hidden_layer_sizes", None)
        architecture = f", camadas {layers}" if layers is not None else ""
        return f"MLP — HOG{architecture}, {negatives} negativos"
    estimator = getattr(classifier, "named_steps", {}).get("model")
    regularization = getattr(estimator, "C", metadata.get("regularization_c", "não informado"))
    suffix = f" + {hard_negatives} negativos difíceis" if hard_negatives else ""
    return f"Regressão — {negatives} negativos{suffix} (C={regularization})"


def run_comparison(
    image: np.ndarray,
    classifiers: dict[str, object],
    thresholds: dict[str, float],
    step: int = 32,
    load_errors: dict[str, str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ComparisonResult:
    """Executa todos os modelos e isola falhas sem interromper a comparação.

    Args:
        image: Mesma imagem BGR fornecida a todos os classificadores.
        classifiers: Modelos restaurados com sucesso, identificados pelo arquivo.
        thresholds: Limiar individual de cada modelo carregado.
        step: Passo máximo das janelas de busca.
        load_errors: Falhas no carregamento para incluir explicitamente no resultado.
        progress: Callback opcional com modelos concluídos e quantidade total.

    Returns:
        Previsões independentes e tempo total, incluindo HOG compartilhado por grupo.
    """
    if classifiers.keys() != thresholds.keys():
        raise ValueError("Informe um limiar para cada modelo carregado.")
    if step < 1 or not np.isfinite(list(thresholds.values())).all():
        raise ValueError("Informe passo positivo e limiares finitos.")
    started = perf_counter()
    groups: dict[DetectorConfig, dict[str, object]] = {}
    results = {
        name: ModelResult(name, model_label(name), None, [], error)
        for name, error in (load_errors or {}).items()
    }
    for name, classifier in classifiers.items():
        config = getattr(classifier, "detector_config_", DetectorConfig())
        groups.setdefault(config, {})[name] = classifier
    total = len(classifiers) + len(load_errors or {})
    for group in groups.values():
        try:
            predictions = detect_many(
                image, group, {name: thresholds[name] for name in group}, step=step
            )
        except Exception:
            # Se um modelo falhar no lote, preserva previsões dos outros integrantes.
            for name, classifier in group.items():
                try:
                    detections = detect(image, classifier, thresholds[name], step=step)
                    results[name] = ModelResult(
                        name, model_label(name, classifier), thresholds[name], detections
                    )
                except Exception as error:
                    results[name] = ModelResult(
                        name, model_label(name, classifier), thresholds[name], [], str(error)
                    )
        else:
            for name, classifier in group.items():
                results[name] = ModelResult(
                    name, model_label(name, classifier), thresholds[name], predictions[name]
                )
        if progress is not None:
            progress(len(results), total)
    return ComparisonResult([results[name] for name in sorted(results)], perf_counter() - started)


@st.cache_resource(show_spinner=False)
def cached_model(name: str, modified_ns: int) -> object:
    """Carrega um modelo e mantém o resultado em cache.

    Args:
        name: Nome do modelo salvo, sem extensão.
        modified_ns: Data de modificação do arquivo, usada para invalidar o cache.

    Returns:
        Modelo ou pipeline treinado, restaurado do pickle local.
    """
    return load_model(name)


def decode_image(content: bytes) -> np.ndarray:
    """Decodifica a imagem enviada para o formato BGR usado pelo detector.

    Args:
        content: Conteúdo binário do arquivo enviado pelo usuário.

    Returns:
        Imagem BGR decodificada pelo OpenCV.
    """
    if not content:
        raise ValueError("O arquivo enviado está vazio.")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Não foi possível ler a imagem. Envie um arquivo JPG, PNG ou WebP válido.")
    return image


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Desenha caixas numeradas nas posições em que o modelo detectou livros.

    Args:
        image: Imagem BGR original.
        detections: Caixas e pontuações retornadas pelo detector.

    Returns:
        Cópia da imagem BGR com as caixas e os rótulos desenhados.
    """
    canvas = image.copy()
    for index, detection in enumerate(detections, start=1):
        x, y, width, height = detection.bbox
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 180, 0), 2)
        cv2.putText(
            canvas,
            f"Livro {index}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def render_model_result(result: ModelResult, image: np.ndarray) -> None:
    """Exibe todas as caixas e o status de uma previsão individual.

    Args:
        result: Detecções ou falha de um dos modelos comparados.
        image: Imagem original BGR, preservada sem alterações.

    Returns:
        Nenhum.
    """
    st.subheader(result.label)
    st.caption(f"Arquivo: {result.name}.pkl")
    if result.error is not None:
        st.error(f"Previsão indisponível: {result.error}")
        return
    st.caption(f"Limiar utilizado: {result.threshold:g}")
    if result.detections:
        st.success(f"Detecção positiva: {len(result.detections)} caixa(s) prevista(s).")
    else:
        st.info("O modelo não detectou livros neste limiar. Isso não garante ausência de livros.")
    st.image(draw_detections(image, result.detections), channels="BGR", width="stretch")
    with st.expander("Coordenadas e pontuações"):
        if result.detections:
            st.dataframe(
                [
                    {
                        "Caixa": index,
                        "X": item.bbox[0],
                        "Y": item.bbox[1],
                        "Largura": item.bbox[2],
                        "Altura": item.bbox[3],
                        "Pontuação": item.score,
                    }
                    for index, item in enumerate(result.detections, start=1)
                ],
                hide_index=True,
            )
        else:
            st.write("Nenhuma caixa prevista.")


def main() -> None:
    """Compara todos os modelos salvos após um único envio de imagem.

    Args:
        Nenhum.

    Returns:
        Nenhum.
    """
    st.set_page_config(page_title="Detecção de livros", page_icon="📚", layout="wide")
    st.title("Detecção de livros")
    st.write("Envie uma imagem para comparar as previsões de todos os modelos salvos.")
    st.caption(
        "As previsões são independentes, não uma votação entre modelos. Caixas são hipóteses: "
        "sem anotações reais, esta tela não calcula precisão, recall ou TP/FP."
    )

    names = sorted(
        path.stem
        for path in MODELS_DIR.glob("*.pkl")
        if path.is_file() and model_family(path.stem) in MODEL_LABELS
    )
    if not names:
        st.info(
            "Nenhum modelo salvo foi encontrado. Salve os modelos do notebook em models/ "
            "e atualize esta página."
        )
        return

    classifiers: dict[str, object] = {}
    thresholds: dict[str, float] = {}
    load_errors: dict[str, str] = {}
    revisions: dict[str, int] = {}
    st.sidebar.header("Configuração da comparação")
    st.sidebar.caption("Cada modelo começa com seu próprio limiar salvo. Nenhum treino é repetido.")
    with st.sidebar.expander("Limiares individuais", expanded=False):
        for name in names:
            try:
                revisions[name] = model_path(name).stat().st_mtime_ns
                classifier = cached_model(name, revisions[name])
                classifiers[name] = classifier
                default_threshold = float(getattr(classifier, "score_threshold_", 0.0))
            except Exception as error:
                load_errors[name] = str(error)
                st.error(f"{model_label(name)}: não foi possível carregar o arquivo.")
                continue
            is_mlp = model_family(name) == "mlp"
            thresholds[name] = st.number_input(
                model_label(name, classifier),
                value=default_threshold,
                step=0.001 if is_mlp else 0.1,
                format="%.3f" if is_mlp else "%.2f",
                key=f"threshold_{name}_{revisions[name]}",
            )
            st.caption(f"Arquivo: {name}.pkl")
            if is_mlp:
                st.caption(f"Probabilidade mínima: {thresholds[name] + 0.5:.1%}.")
            elif not hasattr(classifier, "score_threshold_"):
                st.caption("Sem limiar calibrado salvo; usa zero como padrão.")
    step = st.sidebar.select_slider("Passo da busca (pixels)", options=[16, 32, 64, 96], value=32)
    st.sidebar.caption("Um passo maior acelera a busca, mas pode deixar livros sem detecção.")
    st.sidebar.caption("Margens da regressão e probabilidades do MLP não são comparáveis entre si.")

    st.subheader("Modelos disponíveis para comparação")
    st.dataframe(
        [
            {
                "Experimento": model_label(name, classifiers.get(name)),
                "Arquivo": f"{name}.pkl",
                "Limiar": f"{thresholds[name]:g}" if name in thresholds else "—",
                "Status": "Erro de carregamento" if name in load_errors else "Pronto",
            }
            for name in names
        ],
        hide_index=True,
    )
    if "logistic_regression_002" in names and "logistic_regression_003" in names:
        st.caption(
            "A referência reavaliada usa os mesmos pesos da referência de 30 negativos. "
            "Com o mesmo limiar e passo, suas previsões devem ser iguais."
        )
    if "logistic_regression" in names:
        st.caption("O baseline inicial não tem configuração de janelas salva; usa a busca padrão.")

    upload = st.file_uploader("Imagem", type=["jpg", "jpeg", "png", "webp"])
    if upload is None:
        st.session_state.pop("model_comparison", None)
        return
    try:
        image = decode_image(upload.getvalue())
    except (ValueError, cv2.error) as error:
        st.session_state.pop("model_comparison", None)
        st.error(str(error))
        return

    st.image(image, channels="BGR", caption="Imagem enviada", width="stretch")
    signature = (
        sha256(upload.getvalue()).hexdigest(),
        step,
        tuple(
            (name, revisions.get(name), thresholds.get(name), load_errors.get(name))
            for name in names
        ),
    )
    previous = st.session_state.get("model_comparison")
    if previous is not None and previous[0] != signature:
        st.session_state.pop("model_comparison", None)
        st.info("A imagem, os arquivos ou os parâmetros mudaram. Execute uma nova comparação.")
    if st.button("Comparar todos os modelos", type="primary"):
        st.session_state.pop("model_comparison", None)
        try:
            with st.spinner("Analisando a mesma imagem com todos os modelos..."):
                comparison = run_comparison(image, classifiers, thresholds, step, load_errors)
        except Exception as error:
            st.error(f"Não foi possível executar a comparação: {error}")
            return
        st.session_state["model_comparison"] = (signature, comparison)
    stored = st.session_state.get("model_comparison")
    if stored is None:
        return
    comparison = stored[1]
    st.subheader("Resumo das previsões")
    st.caption(
        f"Tempo total de inferência: {comparison.elapsed_seconds:.1f} s. "
        "HOG é compartilhado entre modelos com as mesmas janelas; este não é um tempo por modelo."
    )
    st.dataframe(
        [
            {
                "Experimento": result.label,
                "Arquivo": f"{result.name}.pkl",
                "Limiar": f"{result.threshold:g}" if result.threshold is not None else "—",
                "Detecção de livro": (
                    "Indisponível"
                    if result.error is not None
                    else "Sim"
                    if result.detections
                    else "Não"
                ),
                "Caixas previstas": len(result.detections) if result.error is None else None,
                "Status": "Erro" if result.error is not None else "Concluído",
            }
            for result in comparison.models
        ],
        hide_index=True,
    )
    for index, result in enumerate(comparison.models):
        if index % 2 == 0:
            columns = st.columns(2)
        with columns[index % 2]:
            render_model_result(result, image)
