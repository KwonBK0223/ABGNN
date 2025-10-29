import os
import json
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_node_mapping(mapping: dict, path: str = "node_mapping.json"):
    payload = {"schema": {"key": "str", "value": "int"}, "data": mapping}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=2)


def load_node_mapping(path: str = "node_mapping.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["data"]
