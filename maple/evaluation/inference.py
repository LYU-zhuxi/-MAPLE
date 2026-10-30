from pathlib import Path
import torch
from ..models.experts import StaticExpert, SynergyExpert
from ..models.router import SparseRouter
from ..data.cache import subset, fingerprint, cache_branches
from ..utils.common import file_hash, load_checkpoint

@torch.no_grad()
def predict_branch(model, encoder, loader, device, progress=None):
    model.eval()
    outputs, representations = [], []
    for records, signals, ctc in loader:
        states, lengths = encoder(signals)
        logits, representation = model(states, lengths, ctc.to(device) if ctc is not None else None)
        outputs.append(logits.cpu())
        representations.append(representation.cpu())
        if progress is not None:
            progress.update()
    return torch.cat(outputs), torch.cat(representations)

def expert_forward(model, payload, kind):
    suffix = "logits" if kind == "static" else "repr"
    return model(payload[f"audio_{suffix}"], payload[f"video_{suffix}"], payload["video_available"])

@torch.no_grad()
def predict_expert(model, payload, kind, device, progress=None):
    model.eval()
    outputs = []
    for indices in torch.arange(len(payload["records"])).split(4096):
        outputs.append(expert_forward(model, subset(payload, indices, device), kind).cpu())
        if progress is not None:
            progress.update()
    return torch.cat(outputs)

def expert_probabilities(payload, checkpoints, device, weights=None):
    result = dict(records=payload["records"], provenance=dict(branches=payload["provenance"]))
    classes = payload["classes"]
    result["classes"] = classes
    if weights is None:
        weights = {kind: load_checkpoint(path, classes) for kind, path in checkpoints.items()}
    for kind in ("static", "synergy"):
        checkpoint = weights[kind]
        model = (StaticExpert() if kind == "static" else SynergyExpert(len(classes))).to(device)
        model.load_state_dict(checkpoint["state"])
        result[kind] = predict_expert(model, payload, kind, device)
        result["provenance"][kind] = dict(sha256=file_hash(checkpoints[kind]), trained_on=checkpoint["train_patients"])
    result["fingerprint"] = fingerprint(result["records"], result["provenance"])
    return result

@torch.no_grad()
def predict_router(model, payload, device, progress=None):
    model.eval()
    probabilities, weights = [], []
    for indices in torch.arange(len(payload["records"])).split(4096):
        out = model(payload["static"][indices].to(device), payload["synergy"][indices].to(device))
        probabilities.append(out["prob"].cpu())
        weights.append(out["weights"].cpu())
        if progress is not None:
            progress.update()
    return torch.cat(probabilities), torch.cat(weights)

def predict(records, config, output, device="cuda:0", weights=None):
    from .evaluate import save_predictions
    root, output = Path(config["paths"]["run"]), Path(output)
    source = Path(config["paths"].get("frozen_run") or root)
    if weights is None:
        paths = {stage: source/stage/"best.pt" for stage in ("audio", "video", "static", "synergy")}
        paths["router"] = root/"router/best.pt"
        weights = {stage: load_checkpoint(path, config["task"]["classes"]) for stage, path in paths.items()}
    cache = cache_branches(dict(predict=records), {m:source/m/"best.pt" for m in ("audio", "video")},
                           config, output/"cache", device, weights=weights)["predict"]
    experts = expert_probabilities(cache, {k:source/k/"best.pt" for k in ("static", "synergy")}, device, weights=weights)
    checkpoint = weights["router"]
    router = SparseRouter(checkpoint["settings"]["fusion"], len(config["task"]["classes"])).to(device)
    router.load_state_dict(checkpoint["state"])
    probabilities, _ = predict_router(router, experts, device)
    save_predictions(output/"predictions.csv", records, probabilities)
    return probabilities
