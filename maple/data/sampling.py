import random
from collections import defaultdict
import torch

def patient_groups(speakers):
    groups = defaultdict(list)
    for i, speaker in enumerate(speakers):
        groups[str(speaker)].append(i)
    return groups

def sample_bags(speakers, rng):
    selected = []
    for _, rows in sorted(patient_groups(speakers).items()):
        selected.extend(rng.sample(rows, 16) if len(rows) >= 16 else rng.choices(rows, k=16))
    rng.shuffle(selected)
    return torch.tensor(selected)

def labels_speakers(records, device="cpu"):
    return torch.tensor([r["label"] for r in records], device=device), [r["speaker"] for r in records]
