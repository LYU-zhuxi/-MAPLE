import math
import torch
from torch import nn
from torch.nn import functional as F


def entmax15(inputs):
    x = inputs / 2
    ordered = x.sort(-1, descending=True).values
    rho = torch.arange(1, x.shape[-1]+1, device=x.device, dtype=x.dtype)
    mean = ordered.cumsum(-1) / rho
    variance = rho * (ordered.square().cumsum(-1)/rho - mean.square())
    threshold = mean - ((1-variance).clamp_min(0)/rho).sqrt()
    support = (threshold <= ordered).sum(-1, keepdim=True).clamp_min(1)
    p = (x - threshold.gather(-1, support-1)).clamp_min(0).square()
    return p / p.sum(-1, keepdim=True).clamp_min(1e-12)


def decision_evidence(s, j):
    def entropy(p):
        p = p.clamp_min(1e-8)
        return -(p*p.log()).sum(-1)/math.log(p.shape[-1])
    def margin(p):
        top = p.topk(2, -1).values
        return top[:, 0] - top[:, 1]
    midpoint = ((s+j)/2).clamp_min(1e-8).log()
    js = .5*((s*(s.clamp_min(1e-8).log()-midpoint)).sum(-1)
             +(j*(j.clamp_min(1e-8).log()-midpoint)).sum(-1))
    classes = s.shape[-1]
    gap = ((j-s)*torch.arange(classes, device=s.device, dtype=s.dtype)).sum(-1)/(classes-1)
    scalars = torch.stack([entropy(s), entropy(j), margin(s), margin(j), js, gap, gap.abs()], -1)
    return torch.cat([s, j, (s-j).abs(), scalars], -1)  # 3*C+7; C=4 gives 19.


def temperature_scale(prob, temperature):
    return (prob.clamp_min(1e-8).log()/temperature).softmax(-1)


def fit_temperature(prob, labels):
    # Fit ONLY training probabilities. Never use validation/test labels here.
    prob, labels = prob.detach().float().cpu(), labels.cpu()
    raw = torch.tensor(math.log(.2/.8), requires_grad=True)
    opt = torch.optim.LBFGS([raw], lr=.25, max_iter=80, tolerance_grad=1e-9,
                          tolerance_change=1e-10, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = F.nll_loss(F.log_softmax(prob.clamp_min(1e-8).log()/(.25+3.75*raw.sigmoid()), -1), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return (.25+3.75*raw.sigmoid()).detach()


class SparseRouter(nn.Module):
    def __init__(self, fusion="entmax", num_classes=4):
        super().__init__()
        if fusion not in ("entmax", "softmax", "average", "global"):
            raise ValueError(f"Unknown fusion: {fusion}")
        self.fusion = fusion
        self.register_buffer("temperatures", torch.ones(2))
        if fusion in ("average", "global"):
            self.logit = nn.Parameter(torch.zeros(()), requires_grad=fusion == "global")
        else:
            self.gate = nn.Sequential(nn.LayerNorm(3*num_classes+7), nn.Linear(3*num_classes+7, 32), nn.GELU(),
                                      nn.Dropout(.1), nn.Linear(32, 2))
            nn.init.zeros_(self.gate[-1].bias)

    def forward(self, static_prob, synergy_prob):
        s = temperature_scale(static_prob, self.temperatures[0])
        j = temperature_scale(synergy_prob, self.temperatures[1])
        if self.fusion in ("average", "global"):
            scores = torch.stack([self.logit*0, self.logit]).expand(s.shape[0], -1)
        else:
            scores = self.gate(decision_evidence(s, j))
        weights = entmax15(scores) if self.fusion == "entmax" else scores.softmax(-1)
        fused = weights[:, :1]*s + weights[:, 1:]*j
        return dict(prob=fused/fused.sum(-1, keepdim=True).clamp_min(1e-8),
                    scores=scores, weights=weights, static=s, synergy=j)
