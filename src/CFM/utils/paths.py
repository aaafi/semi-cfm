import os
from pathlib import Path

def get_project_root() -> Path:
    """
    Return the project root directory by walking up from this file.
    Assumes this file lives in: src/sixgman/utils/
    """
    try:
        return Path(__file__).resolve().parents[3]  # From utils → sixgman → src → project_root
    except NameError:
        # Likely in a notebook
        return Path(os.getcwd()).resolve().parent