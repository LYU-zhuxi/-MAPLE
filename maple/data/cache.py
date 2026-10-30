import hashlib
import json
from pathlib import Path
import torch
from ..models.backbone import PathologyBranch
from ..models.encoders import build_encoder
from ..utils.common import file_hash, load_checkpoint

def fingerprint(records, extra=None):
    content = json.dumps(dict(records=records, extra=extra), sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()

def ctc_key(record):
    return fingerprint([dict(audio=record["audio"], text=record.get("text", ""))])

def cache_branches(splits, checkpoints, config, output, device, weights=None):
    """Recompute explicitly; caches are never selected by a historical version name."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from ..evaluation.inference import predict_branch
    from .dataset import raw_loader
    payloads = {name:dict(records=records) for name, records in splits.items()}
    provenance = {}
    if weights is None:
        weights = {modality: load_checkpoint(path, config["task"]["classes"])
                   for modality, path in checkpoints.items()}
    for modality in ("audio", "video"):
        checkpoint = weights[modality]
        encoder = build_encoder(modality, config["pretrained"], device)
        model = PathologyBranch(**checkpoint["spec"]).to(device)
        model.load_state_dict(checkpoint["state"])
        ctc_dir = config["paths"]["ctc"] if checkpoint["spec"]["dolphin"] else None
        if ctc_dir:
            from ..models.dolphin import validate_cache
            validate_cache(config["pretrained"]["dolphin"], ctc_dir)
        for name, records in splits.items():
            # The historical fusion cache used batch=8 for BOTH branches.
            logits, representation = predict_branch(model, encoder,
                raw_loader(records, modality, ctc_dir, batch_size=8, workers=0), device)
            payloads[name][f"{modality}_logits"] = logits
            payloads[name][f"{modality}_repr"] = representation
        provenance[modality] = dict(sha256=file_hash(checkpoints[modality]),
                                   trained_on=checkpoint["train_patients"])
        del model, encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for name, payload in payloads.items():
        payload["classes"] = config["task"]["classes"]
        payload["video_available"] = torch.ones(len(payload["records"]))
        payload["provenance"] = provenance
        payload["fingerprint"] = fingerprint(payload["records"], provenance)
        torch.save(payload, output/f"{name}.pt")
    return payloads

def subset(payload, indices, device):
    return {key:value[indices].to(device) if torch.is_tensor(value) else
            [value[i] for i in indices.tolist()] if key == "records" else value
            for key, value in payload.items()}
