# MAPLE

This is the official repository for the paper:

> MAPLE: Multi-layer pathology modeling and sparse expert fusion for audio-visual dysarthria severity assessment.
> [![Arxiv](https://img.shields.io/badge/Arxiv--b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/)
> [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/Lyu-1211/maple)

[Overview](#-overview) | [To-Do List](#-to-do-list) | [Quick Start](#-quick-start)

## 🔍 Overview

MAPLE combines multi-layer audio and visual representations with two complementary experts: a static expert for calibrated decision fusion and a synergy expert for cross-modal feature interaction. A sparse router combines their predictions to assess four severity levels.

<p align="center">
  <a href="">
    <img src="/pic/Fig1.png" alt="Logo" width="100%">
  </a>
</p>

## 📝 To-Do List

- [x] Submitted to ICASSP 2027
- [ ] Release project paper
- [ ] Release code
- [x] Release pretrained models

## 🚀 Quick Start

### Create Environment

```bash
# Tested on Ubuntu 22.04 with NVIDIA GPUs (PyTorch CUDA 11.8 wheels).
# System prerequisites: a C++ compiler and audio/video runtime libraries.
# sudo apt-get update
# sudo apt-get install -y build-essential libsndfile1 ffmpeg libgl1 libglib2.0-0
conda create -n maple python=3.9 -y
conda activate maple

# Get the Code
git clone https://github.com/LYU-zhuxi/-MAPLE.git
cd -MAPLE

# Prepare the environment.
pip install pip==26.0.1 setuptools==80.9.0 wheel==0.45.1 Cython==3.2.4
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Skip unused legacy CUDA extensions and preserve the pinned dependencies.
# Regenerate Cython files against the installed NumPy, including after a failed install.
python -m cython -3 --cplus -f \
  external/av_hubert/fairseq/fairseq/data/data_utils_fast.pyx \
  external/av_hubert/fairseq/fairseq/data/token_block_utils_fast.pyx
env -u CUDA_HOME MAX_JOBS=1 python -m pip install \
  --no-build-isolation --no-deps -e external/av_hubert/fairseq
python -m pip install --no-build-isolation --no-deps -e .
```

### Download Pretrained Models

1. **[Qwen3-Omni Audio Encoder](https://huggingface.co/Atotti/Qwen3-Omni-AudioTransformer/tree/main)**: the extracted audio encoder; the full Qwen3-Omni model is not required.
2. **[AV-HuBERT Large](https://facebookresearch.github.io/av_hubert/)**: `large_vox_iter5.pt`, pretrained on LRS3 + VoxCeleb2 without fine-tuning.
3. **[Dolphin-small](https://huggingface.co/DataoceanAI/dolphin-small/tree/main)**: `small.pt` and its configuration, tokenizer, and normalization files.

Run the script below to download all three resources to the paths in `configs/maple.yml`.
```bash
# Run from the repository root in the maple environment.
python scripts/download_pretrained.py
# Downloaded files:
# weights/
# |-- Qwen3-Omni-AudioTransformer/
# |   |-- config.json
# |   |-- preprocessor_config.json
# |   `-- model.safetensors
# |-- avhubert/
# |   `-- large_vox_iter5.pt
# `-- dolphin/
#     |-- small.pt
#     |-- config.yaml
#     |-- bpe.model
#     `-- feats_stats.npz
```

### Download Our Models

Download our trained checkpoints from [Hugging Face](https://huggingface.co/Lyu-1211/maple) and place them in `checkpoints/`:

```bash
# checkpoints/maple_best/
# |-- audio/best.pt
# |-- video/best.pt
# |-- static/best.pt
# |-- synergy/best.pt
# `-- router/best.pt
```

### Inference

Run inference on the examples in `datasets/examples` using our trained checkpoints:

```bash
python infer_demo.py
# Use custom data and checkpoints:
# Segment-level and patient-level predictions are saved to `runs/infer/<checkpoint-folder-name>/`.
# python infer_demo.py --input datasets/my_examples --checkpoints runs/train/my_experiment --device cuda:0
```

For your own data, organize paired audio and mouth-ROI video segments by patient ID and provide a `test.jsonl` file:

```text
datasets/my_examples/
|-- test.jsonl
`-- N_F_10002/
    |-- segment_001.wav
    `-- segment_001.avi
```

Media paths are relative to the input directory. Each `txt` must contain a nonempty spoken transcript; `NA` is not accepted for the demo. Severity labels are not required.

```jsonl
{"ID": "segment_001", "speaker": "N_F_10002", "wav": "N_F_10002/segment_001.wav", "video": "N_F_10002/segment_001.avi", "txt": "冰"}
```

## ⚙️ Training

### Data preparation

Obtain MSDM through the [official MDSA Challenge website](https://huanraozhineng1.github.io/MSDM/), which links to the [Tianchi platform](https://tianchi.aliyun.com/competition/entrance/532191). For access requests, contact the organizers listed on the official website. We do not redistribute the dataset.

Place `MSDM.zip` (containing `Classification_Label.zip` and `Speech_Recognition_Task.zip`) in `datasets/`, then run:

```bash
# datasets/
# `-- MSDM.zip
python scripts/prepare_msdm.py
```

The script extracts the data to `datasets/MSDM/` and generates the fixed Train/Val/Test manifests in `datasets/MSDM/manifests/`.

### Train the unimodal branches

Run the following commands from the repository root. Set `paths.experiment` in `configs/maple.yml` to name your run; use a new name for each experiment. Branch training settings are in `configs/training.yml`. Complete ablation recipes for Tables 2 and 3 are listed as comments at the end of `configs/maple.yml`.

```bash
python -m maple.cli check-data
python -m maple.cli ctc --device cuda:0
python -m maple.cli branch --modality audio --device cuda:0
python -m maple.cli branch --modality video --device cuda:0
```

The `ctc` step prepares Dolphin features for the audio branch. Checkpoints and logs are saved under `runs/train/<experiment>/audio/` and `video/`.

### Train the experts

Train the static and synergy experts using features and predictions from the frozen branches:

```bash
python -m maple.cli experts --device cuda:0
```

The two experts are trained separately and saved as `static/best.pt` and `synergy/best.pt` under the same run directory.

### Train the sparse router

With the default `router.protocol: oof`, first generate patient-level five-fold out-of-fold predictions, then train the router:

```bash
python -m maple.cli cross-fit --device cuda:0
python -m maple.cli router --device cuda:0
```

Cross-fitting retrains both branches and experts within each fold and predicts the held-out patients. It is the most time-consuming stage. The merged predictions are saved to `oof/oof.pt`; the router checkpoint is saved to `router/best.pt`. Final inference uses the experts trained on the full training split, not the fold-specific experts.

For router-only ablations, set `paths.frozen_run` to `runs/train/<source_experiment>` to reuse its branches, experts, and prediction caches. Choose a new `paths.experiment`, apply a Table 3 recipe, and run only `router` and `test`. OOF variants require the source run's completed cross-fitting results.

### Evaluation

Evaluate the complete model on the test split:

```bash
python -m maple.cli test --device cuda:0

# Optional: evaluate individual branches and experts.
python -m maple.cli test-branch --modality audio --device cuda:0
python -m maple.cli test-branch --modality video --device cuda:0
python -m maple.cli test-experts --device cuda:0
```

Results are saved to `runs/test/<experiment>/`, including predictions and segment-level Precision, Accuracy, F1, patient-level F1, and Final F1. All commands use `configs/maple.yml` by default; add `--config path/to/config.yml` to use another configuration.

## Citation

```

```
