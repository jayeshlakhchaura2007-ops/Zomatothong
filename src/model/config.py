"""Load model config from YAML."""
from pathlib import Path
from typing import Any

import yaml


def load_model_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "model.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
