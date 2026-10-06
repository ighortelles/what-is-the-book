"""Entrada da aplicação: execute com streamlit run src/app/main.py."""

import sys
from pathlib import Path

# Permite importar src quando o Streamlit executa este arquivo diretamente.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.app.interface import main  # noqa: E402

if __name__ == "__main__":
    main()
