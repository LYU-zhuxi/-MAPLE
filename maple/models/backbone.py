"""Task backbone shared by audio and video; pretrained encoders are separate."""
import torch
from torch import nn
from torch.nn import functional as F


def valid_mask(lengths, time):
    return torch.arange(time, device=lengths.device)[None] < lengths[:, None]


class ConformerLite(nn.Module):
    def __init__(self):
        super().__init__()
        def ffn():
            return nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512), nn.SiLU(),
                                 nn.Dropout(.1), nn.Linear(512, 256), nn.Dropout(.1))
        self.ffn1 = ffn()
        self.attn_norm = nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(256, 4, dropout=.1, batch_first=True)
        self.attn_dropout = nn.Dropout(.1)
        self.conv_norm = nn.LayerNorm(256)
        self.pointwise_in = nn.Conv1d(256, 512, 1)
        self.depthwise = nn.Conv1d(256, 256, 15, padding=7, groups=256)
        self.group_norm = nn.GroupNorm(1, 256)
        self.pointwise_out = nn.Conv1d(256, 256, 1)
        self.conv_dropout = nn.Dropout(.1)
        self.ffn2 = ffn()
        self.final_norm = nn.LayerNorm(256)

    def forward(self, x, lengths):
        mask = valid_mask(lengths, x.shape[1])
        m = mask.unsqueeze(-1).to(x.dtype)
        x = x * m
        x = (x + .5 * self.ffn1(x) * m) * m
        a = self.attn_norm(x)
        a = self.attn(a, a, a, key_padding_mask=~mask, need_weights=False)[0]
        x = (x + self.attn_dropout(a)) * m
        c = (self.conv_norm(x) * m).transpose(1, 2)
        c = F.glu(self.pointwise_in(c), dim=1)
        c = F.silu(self.group_norm(self.depthwise(c)))
        c = self.conv_dropout(self.pointwise_out(c).transpose(1, 2)) * m
        x = (x + c) * m
        return self.final_norm(x + .5 * self.ffn2(x) * m) * m


class LayerAdapter(nn.Module):
    def __init__(self, encoder_dim):
        super().__init__()
        self.projection = nn.Linear(encoder_dim, 256)
        self.temporal = ConformerLite()

    def forward(self, x, lengths):
        return self.temporal(self.projection(x), lengths)


class LayerFusion(nn.Module):
    def __init__(self, layers=3, mode="local", chunk=4):
        super().__init__()
        self.layers, self.mode, self.chunk = layers, mode, chunk
        if mode == "local":
            self.gate = nn.Sequential(nn.LayerNorm(layers * 256),
                nn.Linear(layers * 256, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, layers))
        elif mode == "global":
            self.layer_logits = nn.Parameter(torch.linspace(-.4, .4, layers))
        elif mode == "concat":
            self.projection = nn.Linear(layers * 256, 256)
        elif mode != "single":
            raise ValueError(f"Unknown paper ablation: {mode}")

    def forward(self, states, lengths):
        # Three independent [B,T,256] streams -> one [B,T,256] stream.
        b, t, d = states[0].shape
        mask = valid_mask(lengths, t).unsqueeze(-1)
        if self.mode == "single":
            return states[0] * mask
        if self.mode == "concat":
            return self.projection(torch.cat(states, -1)) * mask
        if self.mode == "global":
            weights = self.layer_logits.softmax(0)[None, None, :, None]
        else:
            padding = (-t) % self.chunk
            cm = F.pad(mask, (0, 0, 0, padding)).reshape(b, -1, self.chunk, 1)
            summaries = []
            for h in states:
                h = F.pad(h, (0, 0, 0, padding)).reshape(b, -1, self.chunk, d)
                summaries.append((h * cm).sum(2) / cm.sum(2).clamp_min(1e-6))
            # Masked chunk means, concatenated over depth: [B,N,3*256].
            weights = self.gate(torch.cat(summaries, -1)).softmax(-1)
            weights = weights * cm.any(2)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            weights = weights.repeat_interleave(self.chunk, 1)[:, :t, :, None]
        return (torch.stack(states, 2) * weights).sum(2) * mask


def stats_pool(x, lengths):
    mask = valid_mask(lengths, x.shape[1]).unsqueeze(-1).to(x.dtype)
    count = mask.sum(1).clamp_min(1)
    mean = (x * mask).sum(1) / count
    variance = (((x - mean[:, None]) * mask) ** 2).sum(1) / count
    # Population standard deviation, matching the training source (eps=1e-6).
    return torch.cat([mean, (variance.clamp_min(0) + 1e-6).sqrt()], -1)


class CTCResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Sequential(nn.LayerNorm(15), nn.Linear(15, 64), nn.GELU(),
            nn.Dropout(.2), nn.Linear(64, 32), nn.GELU())
        def head(out):
            return nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 64), nn.GELU(),
                                 nn.Dropout(.1), nn.Linear(64, out))
        self.delta, self.gate = head(512), head(1)
        nn.init.constant_(self.gate[-1].bias, -3)

    def forward(self, representation, ctc):
        z = self.projector(ctc)
        # R_audio = R + 0.05 * sigmoid(g(z)) * tanh(delta(z)); R has 512 channels.
        return representation + .05 * self.gate(z).sigmoid() * self.delta(z).tanh()


class PathologyBranch(nn.Module):
    def __init__(self, encoder_dim, fusion="local", chunk=4, dolphin=False, num_classes=4):
        super().__init__()
        self.spec = dict(encoder_dim=encoder_dim, fusion=fusion, chunk=chunk, dolphin=dolphin, num_classes=num_classes)
        self.adapters = nn.ModuleList([LayerAdapter(encoder_dim)
                                      for _ in range(1 if fusion == "single" else 3)])
        self.fusion = LayerFusion(len(self.adapters), fusion, chunk)
        self.post = ConformerLite()
        self.ctc = CTCResidual() if dolphin else None
        self.classifier = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256), nn.GELU(),
            nn.Dropout(.2), nn.Linear(256, 128), nn.GELU(), nn.Dropout(.2), nn.Linear(128, num_classes))

    def forward(self, states, lengths, ctc=None):
        states = states[-1:] if self.spec["fusion"] == "single" else states
        h = [adapter(x, lengths) for adapter, x in zip(self.adapters, states)]
        representation = stats_pool(self.post(self.fusion(h, lengths), lengths), lengths)
        if self.ctc is not None:
            representation = self.ctc(representation, ctc)
        return self.classifier(representation), representation
