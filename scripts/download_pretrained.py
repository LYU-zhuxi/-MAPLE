"""Download the frozen pretrained models used by configs/maple.yml."""
from pathlib import Path

from huggingface_hub import snapshot_download
from torch.hub import download_url_to_file


def download_models(root):
    weights = Path(root) / "weights"
    models = [
        ("Atotti/Qwen3-Omni-AudioTransformer", "Qwen3-Omni-AudioTransformer",
         ["config.json", "preprocessor_config.json", "model.safetensors"]),
        ("DataoceanAI/dolphin-small", "dolphin",
         ["small.pt", "config.yaml", "bpe.model", "feats_stats.npz"]),
    ]
    for repo, directory, files in models:
        print(f"Downloading {repo} ...", flush=True)
        snapshot_download(repo, local_dir=str(weights / directory), allow_patterns=files)

    checkpoint = weights / "avhubert" / "large_vox_iter5.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        print(f"Keeping existing {checkpoint}", flush=True)
    else:
        print("Downloading AV-HuBERT Large ...", flush=True)
        # torch.hub downloads to a temporary file before moving it into place.
        download_url_to_file(
            "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/"
            "clean-pretrain/large_vox_iter5.pt",
            str(checkpoint), progress=True,
        )
    print(f"Pretrained models ready in {weights}", flush=True)


if __name__ == "__main__":
    download_models(Path(__file__).resolve().parents[1])
