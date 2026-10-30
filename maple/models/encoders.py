"""Frozen encoder adapters. No pretrained implementation or weights are vendored."""
import importlib
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import valid_mask


def audio_output_lengths(lengths):
    rest = lengths % 100
    rest = (rest-1)//2+1
    return ((rest-1)//2+1-1)//2+1+(lengths//100)*13


class QwenAudio(nn.Module):
    # Zero-based block-output indices, exactly as in the training implementation.
    selected_layers = (11, 19, 27)

    def __init__(self, path, device):
        super().__init__()
        from transformers import WhisperFeatureExtractor
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
        self.features = WhisperFeatureExtractor.from_pretrained(path)
        self.encoder = Qwen3OmniMoeAudioEncoder.from_pretrained(path, torch_dtype=torch.float32).to(device)
        self.encoder.requires_grad_(False).eval()
        self.dim = self.encoder.config.d_model
        self.device = torch.device(device)

    @torch.no_grad()
    def forward(self, waveforms):
        batch = self.features(waveforms, sampling_rate=16000, return_tensors="pt",
                              padding="longest", return_attention_mask=True)
        mask = batch["attention_mask"].bool()
        lengths = mask.sum(-1).to(self.device)
        output_lengths = audio_output_lengths(lengths)
        x = batch["input_features"].permute(0, 2, 1)[mask].T.to(self.device)
        model = self.encoder
        # Reproduce the audio-only encoder's packed convolution/attention path.
        counts = torch.ceil(lengths/(model.n_window*2)).long()
        chunks = torch.full((int(counts.sum()),), model.n_window*2, device=self.device, dtype=torch.long)
        tails = F.pad(counts, (1, 0), value=-1).cumsum(0)[1:]
        chunks[tails] = lengths % (model.n_window*2)
        chunks[chunks == 0] = model.n_window*2
        padded = nn.utils.rnn.pad_sequence(x.T.split(chunks.tolist()), batch_first=True).transpose(1, 2)
        chunk_lengths = audio_output_lengths(chunks)
        chunk_mask = nn.utils.rnn.pad_sequence([torch.ones(int(n), dtype=torch.bool, device=self.device)
                                               for n in chunk_lengths], batch_first=True)
        embeddings = []
        for part in padded.unsqueeze(1).split(model.conv_chunksize):
            embeddings.append(F.gelu(model.conv2d3(F.gelu(model.conv2d2(F.gelu(model.conv2d1(part)))))))
        x = torch.cat(embeddings)
        b, c, f, t = x.shape
        x = model.conv_out(x.permute(0, 3, 1, 2).reshape(b, t, c*f))
        x = x + model.positional_embedding.positional_embedding[:t].unsqueeze(0).to(x.dtype)
        x = x[chunk_mask]
        windows = [0]
        width = chunk_mask.shape[-1]*(model.n_window_infer//(model.n_window*2))
        for n in output_lengths.tolist():
            windows.extend([width]*(n//width))
            if n % width:
                windows.append(n % width)
        cu_seqlens = torch.tensor(windows, device=self.device).cumsum(0, dtype=torch.int32)
        states = []
        for i, layer in enumerate(model.layers):
            x = layer(x, cu_seqlens)[0]
            if i in self.selected_layers:
                states.append(nn.utils.rnn.pad_sequence(x.split(output_lengths.tolist()), batch_first=True))
        return states, output_lengths


class AVHubertVideo(nn.Module):
    # AV-HuBERT uses one-based layer numbers, unlike the audio wrapper above.
    selected_layers = (8, 16, 24)

    def __init__(self, checkpoint, source, device):
        super().__init__()
        root = Path(source).resolve()
        sys.path.insert(0, str(root / "fairseq"))
        sys.path.insert(0, str(root / "avhubert"))
        sys.path.insert(0, str(root))
        # Register AV-HuBERT as a package so its relative imports resolve.
        importlib.import_module("avhubert")
        from fairseq import checkpoint_utils
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)])
        self.encoder = models[0].to(device).requires_grad_(False).eval()
        self.encoder.remove_pretraining_modules()
        self.dim = int(self.encoder.encoder_embed_dim)  # Large checkpoint: 1024, not YAML fallback 768.
        self.device = torch.device(device)

    @torch.no_grad()
    def forward(self, videos):
        lengths = torch.tensor([len(v) for v in videos], device=self.device)
        padded = nn.utils.rnn.pad_sequence(videos, batch_first=True).to(self.device)
        padding_mask = ~valid_mask(lengths, padded.shape[1])
        model = self.encoder
        video = model.forward_features(padded.transpose(1, 2), modality="video")
        audio = video.new_zeros(video.shape[0], model.encoder_embed_dim, video.shape[-1])
        x = torch.cat([audio, video], 1) if model.modality_fuse == "concat" else audio+video
        x = model.layer_norm(x.transpose(1, 2))
        mask = model.forward_padding_mask(x, padding_mask)
        if model.post_extract_proj is not None:
            x = model.post_extract_proj(x)
        _, results = model.encoder.extract_features(model.dropout_input(x), mask, tgt_layer=23)
        states = [results[i-1][0].transpose(0, 1) for i in self.selected_layers]
        if model.encoder.layer_norm_first:
            states[-1] = model.encoder.layer_norm(states[-1])
        return states, (~mask).sum(-1)


def build_encoder(modality, config, device):
    if modality == "audio":
        return QwenAudio(config["qwen"], device)
    return AVHubertVideo(config["avhubert"], config["avhubert_source"], device)
