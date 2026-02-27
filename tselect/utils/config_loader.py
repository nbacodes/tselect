from pathlib import Path
from .loader import load_yaml

def load_tselect_config(repo_root: Path):
    config_path = repo_root / "tselect.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            "tselect.yaml not found in repository root"
        )

    return load_yaml(config_path)
