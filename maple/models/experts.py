import math
import torch
from torch import nn
from torch.nn import functional as F


class StaticExpert(nn.Module):
    def __init__(self):
        super().__init__()
        initial = math.log(math.expm1(.5))
        self.audio_temperature_raw = nn.Parameter(torch.tensor(initial))
        self.video_temperature_raw = nn.Parameter(torch.tensor(initial))
        self.video_weight_logit = nn.Parameter(torch.tensor(0.))

    def forward(self, audio_logits, video_logits, video_available):
        ta = (.5 + F.softplus(self.audio_temperature_raw)).clamp(max=5)
        tv = (.5 + F.softplus(self.video_temperature_raw)).clamp(max=5)
        beta = self.video_weight_logit.sigmoid() * video_available.reshape(-1, 1)
        # Three learned scalars. This is decision-level, not feature-level fusion.
        return ((1 - beta) * audio_logits / ta + beta * video_logits / tv).softmax(-1)


class SynergyExpert(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        def projection():
            return nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 32), nn.GELU(), nn.Dropout(.1))
        self.audio_projection, self.video_projection = projection(), projection()
        self.classifier = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(),
                                        nn.Dropout(.1), nn.Linear(64, num_classes))

    def forward(self, audio_repr, video_repr, video_available):
        a = self.audio_projection(audio_repr)
        v = self.video_projection(video_repr) * video_available.reshape(-1, 1)
        # u=[z_a,z_v,|z_a-z_v|,z_a*z_v]: 4*32=128, no label in forward.
        return self.classifier(torch.cat([a, v, (a-v).abs(), a*v], -1)).softmax(-1)
