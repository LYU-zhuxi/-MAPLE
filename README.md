# MAPLE

This is the official repository for the paper:

> MAPLE: Multi-layer pathology modeling and sparse expert fusion for audio-visual dysarthria severity assessment.
> [![Arxiv](https://img.shields.io/badge/Arxiv--b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/)
> [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/your-model)

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
- [ ] Release pretrained models

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

### Download Pretrained models

1. **[Qwen3-Omni Audio Encoder](https://huggingface.co/Atotti/Qwen3-Omni-AudioTransformer/tree/main)**: the extracted audio encoder; the full Qwen3-Omni model is not required.
2. **[AV-HuBERT Large](https://facebookresearch.github.io/av_hubert/)**: `large_vox_iter5.pt`, pretrained on LRS3 + VoxCeleb2 without fine-tuning.
3. **[Dolphin-small](https://huggingface.co/DataoceanAI/dolphin-small/tree/main)**: `small.pt` and its configuration, tokenizer, and normalization files.

Run the script below to download all three resources to the paths in `configs/maple.yml`.
```bash
# Run from the repository root in the maple environment.
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
python scripts/download_pretrained.py
```

### Running the Demo




### Data preparation



### Training

### 1. Train the unimodal branches


### 2. Train the experts



### 3. Train the sparse router



### Evaluation




## Citation

```

```
