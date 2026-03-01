"""Load feature config from YAML."""
from pathlib import Path
from typing import Any

import yaml


def load_feature_config(config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "features.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
