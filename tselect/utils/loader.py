import yaml
import json
from pathlib import Path


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
