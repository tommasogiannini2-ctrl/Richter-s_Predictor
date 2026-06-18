import os

import yaml


def load_config(path: str | None) -> dict:
    """
    Carica un file YAML di configurazione.

    Se path e' None restituisce un dizionario vuoto, cosi' il chiamante puo'
    mantenere il comportamento interattivo preesistente.
    """
    if path is None:
        return {}

    config_path = os.path.abspath(path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("Il file YAML deve contenere una mappa di configurazione.")

    return config


def get_nested(config: dict, path: str, default=None):
    """Legge una chiave annidata con notazione puntata, ad esempio run.use_saved_model."""
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
