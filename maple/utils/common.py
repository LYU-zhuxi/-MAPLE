import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_checkpoint(path, classes):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != classes:
        raise ValueError(f"Class mapping in {path} differs from task.classes.")
    return checkpoint


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

def log_epoch(path, stage, epoch, losses, validation):
    row = dict(stage=stage, epoch=epoch, train=losses, validation=validation)
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row)+"\n")

def fresh_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "best.pt").exists() or (path / "history.jsonl").exists():
        raise FileExistsError(f"Use a new output directory; refusing to mix training runs: {path}")
    return path
