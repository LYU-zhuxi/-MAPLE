import csv
import json
from pathlib import Path
import numpy as np
import torch
from ..models.backbone import PathologyBranch
from ..models.encoders import build_encoder
from ..models.router import SparseRouter
from ..data.dataset import raw_loader
from ..data.sampling import labels_speakers
from ..utils.common import save_json, load_checkpoint
from .metrics import metrics
from .inference import predict_branch, predict_router, expert_probabilities
from ..data.cache import cache_branches

def save_predictions(path, records, probabilities):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "speaker", "label", "prediction", *[f"p{i}" for i in range(probabilities.shape[-1])]])
        for record, p in zip(records, probabilities.tolist()):
            writer.writerow([record["id"], record["speaker"], record.get("label", ""), int(np.argmax(p)), *p])


def test_final(config, test_records, device, experts_only=False):
    root = Path(config["paths"]["run"])
    source = Path(config["paths"].get("frozen_run") or root)
    output = Path(config["paths"]["results"])
    output.mkdir(parents=True, exist_ok=True)
    branch_checkpoints = {m:source/m/"best.pt" for m in ("audio", "video")}
    cache = cache_branches(dict(test=test_records), branch_checkpoints, config, output/"branch_cache", device)["test"]
    experts = {k:source/k/"best.pt" for k in ("static", "synergy")}
    payload = expert_probabilities(cache, experts, device)
    torch.save(payload, output/"test_experts.pt")
    if experts_only:
        results = {}
        for kind in ("static", "synergy"):
            results[kind] = metrics(np.array([r["label"] for r in test_records]),
                                    payload[kind].argmax(-1).numpy(), [r["speaker"] for r in test_records], payload[kind].shape[-1])
            save_predictions(output/f"{kind}_predictions.csv", test_records, payload[kind])
        save_json(output/"expert_metrics.json", results)
        print(results)
        return results
    return evaluate_router(payload, root/"router/best.pt", config["paths"]["results"], device)

def evaluate_branch(records, modality, config, checkpoint_path, output, device):
    checkpoint = load_checkpoint(checkpoint_path, config["task"]["classes"])
    encoder = build_encoder(modality, config["pretrained"], device)
    model = PathologyBranch(**checkpoint["spec"]).to(device)
    model.load_state_dict(checkpoint["state"])
    ctc_dir = config["paths"]["ctc"] if checkpoint["spec"]["dolphin"] else None
    if ctc_dir:
        from ..models.dolphin import validate_cache
        validate_cache(config["pretrained"]["dolphin"], ctc_dir)
    logits, _ = predict_branch(model, encoder, raw_loader(records, modality, ctc_dir), device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    y, speakers = labels_speakers(records)
    result = metrics(y.numpy(), logits.argmax(-1).numpy(), speakers, logits.shape[-1])
    save_json(output/"metrics.json", result)
    save_predictions(output/"predictions.csv", records, logits.softmax(-1))
    print(json.dumps(result, indent=2))
    return result

def evaluate_router(payload, checkpoint, output, device):
    checkpoint = load_checkpoint(checkpoint, payload["classes"])
    model = SparseRouter(checkpoint["settings"]["fusion"], payload["static"].shape[-1]).to(device)
    model.load_state_dict(checkpoint["state"])
    probabilities, weights = predict_router(model, payload, device)
    y, speakers = labels_speakers(payload["records"])
    result = metrics(y.numpy(), probabilities.argmax(-1).numpy(), speakers, probabilities.shape[-1])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_predictions(output/"predictions.csv", payload["records"], probabilities)
    save_json(output/"metrics.json", result)
    torch.save(dict(weights=weights), output/"routing_weights.pt")
    print(json.dumps(result, indent=2))
    return result
