from pathlib import Path
import json
import yaml


def load_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes = config["task"]["classes"]
    if len(classes) < 2 or set(classes) != set(range(len(classes))) or any(not isinstance(v, str) or not v.strip() for v in classes.values()):
        raise ValueError("task.classes must map consecutive labels 0..C-1 to nonempty severity names.")
    config["task"]["num_classes"] = len(classes)
    training_path = path.parent / config["training_config"]
    # Resolve before saving config.json so every run records its actual settings.
    config["training"] = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    name = paths["experiment"]
    if not isinstance(name, str) or not name.strip() or name in (".", "..") or any(c in name for c in "/\\:"):
        raise ValueError("paths.experiment must be a folder name, not a path.")
    paths["run"] = (Path("runs") / "train" / name).as_posix()
    paths["results"] = paths.get("results") or (Path("runs") / "test" / name).as_posix()
    return config


def run_config_settings(config, command):
    if command in ("test", "test-branch", "test-experts"):
        settings = {key: config[key] for key in ("task", "pretrained")}
    else:
        settings = dict(config, paths={key: value for key, value in config["paths"].items()
                                      if key != "results"})
    # Saved JSON converts integer class keys to strings.
    return json.loads(json.dumps(settings))
