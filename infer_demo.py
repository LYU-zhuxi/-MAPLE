"""Run the MAPLE inference demo with paired examples from test.jsonl."""
import argparse
import csv
from pathlib import Path


def check_inputs(input_dir, checkpoint_dir, classes, dolphin):
    from maple.data.manifest import read_manifest
    from maple.models.dolphin import normalize_text
    manifest = Path(input_dir)/"test.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Example manifest not found: {manifest}")
    records = read_manifest(manifest, labeled=False, data_root=input_dir, num_classes=len(classes), require_text=True)
    if not records:
        raise ValueError(f"No segments in {manifest}")
    missing = []
    for record in records:
        if not normalize_text(record["text"]):
            raise ValueError(f"Demo segment {record['id']} requires txt with valid spoken text; empty/NA references are not supported.")
        for key in ("audio", "video"):
            if not record.get(key) or not Path(record[key]).is_file():
                missing.append(f"{record['id']}: {key}={record.get(key, '<missing>')}")
    for stage in ("audio", "video", "static", "synergy", "router"):
        path = Path(checkpoint_dir)/stage/"best.pt"
        if not path.is_file():
            missing.append(str(path))
    for key in ("checkpoint", "config", "bpe", "stats"):
        path = Path(dolphin[key])
        if not path.is_file():
            missing.append(f"Dolphin {key}: {path}")
    if missing:
        raise FileNotFoundError("Missing media/checkpoints:\n" + "\n".join(missing))
    return records


def save_results(records, probabilities, classes, output):
    from maple.data.sampling import patient_groups
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = probabilities.argmax(-1).tolist()
    with (output/"segments.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["speaker", "id", "prediction", "severity", *[f"p{i}" for i in range(len(classes))]])
        for record, label, prob in zip(records, predictions, probabilities.tolist()):
            writer.writerow([record["speaker"], record["id"], label, classes[label], *prob])
    with (output/"patients.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["speaker", "segments", "prediction", "severity", *[f"votes{i}" for i in range(len(classes))]])
        print(f"{'Patient':<20} {'Segments':>8}  Prediction")
        for speaker, rows in patient_groups([r["speaker"] for r in records]).items():
            votes = [sum(predictions[i] == label for i in rows) for label in range(len(classes))]
            # Same majority vote and lower-index tie rule as paper evaluation.
            label = max(range(len(classes)), key=lambda i: votes[i])
            writer.writerow([speaker, len(rows), label, classes[label], *votes])
            print(f"{speaker:<20} {len(rows):>8}  {classes[label]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/maple.yml")
    parser.add_argument("--input", default="datasets/examples", help="Folder containing test.jsonl and paired media")
    parser.add_argument("--checkpoints", default="checkpoints/maple_best", help="Folder containing stage/best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", help="Default: runs/infer/<checkpoint-folder-name>")
    args = parser.parse_args()
    from maple.utils.config import load_config
    config = load_config(args.config)
    classes = config["task"]["classes"]
    dolphin = config["pretrained"]["dolphin"]
    records = check_inputs(args.input, args.checkpoints, classes, dolphin)
    # File checks finish before any model or checkpoint is loaded.
    from maple.utils.common import load_checkpoint
    root = Path(args.checkpoints)
    audio = load_checkpoint(root/"audio/best.pt", classes)
    if not audio["spec"]["dolphin"]:
        raise ValueError("The demo requires a Dolphin-enabled audio checkpoint; w/o Dolphin checkpoints are for ablation evaluation only.")
    weights = dict(audio=audio)
    weights.update({stage: load_checkpoint(root/stage/"best.pt", classes)
                    for stage in ("video", "static", "synergy", "router")})
    output = Path(args.output) if args.output else Path("runs")/"infer"/root.resolve().name
    config["paths"].update(run=str(root), frozen_run=None, ctc=str(output/"ctc"))
    from maple.models.dolphin import prepare_ctc
    prepare_ctc(records, dolphin, config["paths"]["ctc"], args.device)
    from maple.evaluation.inference import predict
    probabilities = predict(records, config, output, args.device, weights=weights)
    save_results(records, probabilities, classes, output)


if __name__ == "__main__":
    main()
