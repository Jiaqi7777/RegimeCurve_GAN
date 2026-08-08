from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def ensure_output(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_json(value: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")

