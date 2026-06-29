"""File I/O helpers: load documents, save results."""
import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
