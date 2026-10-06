"""Persistência dos modelos treinados em arquivos pickle."""

import pickle
from pathlib import Path

from src.utils.path import model_path


def save_model(model: object, name: str) -> Path:
    """Salva um modelo treinado sem sobrescrever arquivos existentes.

    O primeiro arquivo usa o nome solicitado. Salvamentos posteriores recebem
    sufixos numéricos, como ``mlp_001.pkl`` e ``mlp_002.pkl``.

    Args:
        model: Modelo ou pipeline treinado a ser serializado.
        name: Nome do arquivo, sem diretórios ou extensão.

    Returns:
        Caminho absoluto do novo arquivo pickle salvo.
    """
    content = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    destination = model_path(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    version = 0
    while True:
        destination = model_path(name if version == 0 else f"{name}_{version:03d}")
        try:
            file = destination.open("xb")
        except FileExistsError:
            version += 1
            continue
        try:
            with file:
                file.write(content)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return destination


def load_model(name: str) -> object:
    """Carrega um modelo salvo para reutilização sem novo treinamento.

    Carregue apenas arquivos pickle de origem confiável, no mesmo ambiente
    de dependências usado para salvar o modelo.

    Args:
        name: Nome exato do arquivo salvo, incluindo o sufixo de versão quando
            houver, sem diretórios ou extensão (por exemplo, ``mlp_001``).

    Returns:
        Modelo ou pipeline restaurado, incluindo seus parâmetros aprendidos.
    """
    with model_path(name).open("rb") as file:
        return pickle.load(file)
