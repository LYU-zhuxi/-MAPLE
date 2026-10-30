import json
from pathlib import Path
from .sampling import patient_groups

def read_manifest(path, labeled=True, data_root=None, num_classes=4, require_text=False):
    path = Path(path)
    base = Path(data_root) if data_root is not None else path.parent
    content = path.read_text(encoding="utf-8-sig").strip()
    records = [json.loads(line) for line in content.splitlines() if line.strip()]
    ids = [r.get("id", r.get("ID")) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate segment IDs in {path}")
    for r in records:
        if not ("id" in r or "ID" in r) or "speaker" not in r:
            raise ValueError("Every segment requires ID (or id) and speaker.")
        if require_text and not isinstance(r.get("txt", r.get("text")), str):
            raise ValueError("Every demo segment requires txt containing its spoken text.")
        if "id" not in r and "ID" in r:
            r["id"] = str(r["ID"])
        if "label" in r and (type(r["label"]) is not int or r["label"] not in range(num_classes)):
            raise ValueError(f"Label {r['label']} is outside task.classes (0..{num_classes-1}).")
        if labeled and "label" not in r:
            raise ValueError("Labeled manifests require label in every record.")
        r["speaker"] = str(r["speaker"])
        if "wav" in r:
            r["audio"] = r["wav"]
        r["text"] = r.get("txt", r.get("text", "NA"))
        for key in ("audio", "video"):
            if r.get(key):
                r[key] = str((base / r[key]).resolve())
    for rows in patient_groups([r["speaker"] for r in records]).values():
        if labeled and len({records[i]["label"] for i in rows}) != 1:
            raise ValueError("A patient must have a consistent severity label.")
    return records

def check_splits(splits):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            pa = {r["speaker"] for r in splits[a]}
            pb = {r["speaker"] for r in splits[b]}
            if pa & pb:
                raise ValueError(f"Patient leakage between {a} and {b}.")
            if {r["audio"] for r in splits[a]} & {r["audio"] for r in splits[b]}:
                raise ValueError(f"Audio overlap between {a} and {b}.")
