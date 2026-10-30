"""Fifteen frozen Dolphin-small CTC statistics; no severity label is used."""
import argparse
import math
import unicodedata
from pathlib import Path
import torch
import yaml
from ..data.dataset import load_audio
from ..data.cache import ctc_key


def normalize_text(text):
    # Missing references are not the literal spoken letters "NA".
    if text.strip().lower() in {"", "na", "nan", "none", "null"}:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKC", text).strip()
                   if not c.isspace() and unicodedata.category(c)[0] not in "PS")


def ctc_statistics(log_prob, blank_id, text_nll, prompt_nll):
    p = log_prob.exp()
    entropy = -(p*log_prob).sum(-1)
    blank, top1 = p[:, blank_id], p.max(-1).values
    nonblank = 1-blank
    run, longest = 0, 0
    for flag in (blank >= .8).tolist():
        run = run+1 if flag else 0
        longest = max(longest, run)
    values = [entropy.mean(), entropy.std(unbiased=False), torch.quantile(entropy, .9),
              entropy.topk(max(1, math.ceil(.1*len(entropy)))).values.mean(),
              blank.mean(), blank.std(unbiased=False), (blank >= .8).float().mean(),
              blank.new_tensor(longest/len(blank)), top1.mean(), top1.std(unbiased=False),
              (top1 <= .5).float().mean(), nonblank.mean(), nonblank.std(unbiased=False),
              text_nll, prompt_nll]
    return torch.stack(values).float()


class DolphinStatistics:
    def __init__(self, config, device):
        from espnet2.tasks.s2t import S2TTask
        from espnet2.text.build_tokenizer import build_tokenizer
        from espnet2.text.token_id_converter import TokenIDConverter
        cfg = yaml.safe_load(Path(config["config"]).read_text(encoding="utf-8"))
        # The released generic config needs the exact Dolphin-small dimensions.
        cfg["encoder_conf"].update(output_size=768, attention_heads=12, cgmlp_linear_units=3072,
                                   num_blocks=12, linear_units=1536)
        cfg["decoder_conf"].update(attention_heads=12, linear_units=3072, num_blocks=12)
        cfg["bpemodel"] = config["bpe"]
        cfg.setdefault("preprocessor_conf", {})["bpemodel"] = config["bpe"]
        cfg.setdefault("normalize_conf", {})["stats_file"] = config["stats"]
        self.model = S2TTask.build_model(argparse.Namespace(**cfg))
        state = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        status = self.model.load_state_dict(state, strict=False)
        missing = [k for k in status.missing_keys if k.startswith(("encoder.", "ctc.", "frontend.", "normalize."))]
        if missing:
            raise ValueError(f"Dolphin feature weights missing: {missing[:5]}")
        self.model.to(device=device, dtype=torch.float32).requires_grad_(False).eval()
        self.tokenizer = build_tokenizer(token_type="bpe", bpemodel=config["bpe"])
        self.converter = TokenIDConverter(token_list=self.model.token_list)
        self.blank = next((self.converter.token2id[t] for t in ("<blank>", "<blk>", "<ctc_blank>")
                           if t in self.converter.token2id), 0)
        self.prompt = [self.converter.token2id[t] for t in ("<zh>", "<CN>", "<asr>")]
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, waveform, text):
        wav = torch.as_tensor(waveform[:30*16000], device=self.device).float()
        if wav.abs().max() > 1.5:
            wav = wav/wav.abs().max()
        enc, lengths = self.model.encode(speech=wav[None], speech_lengths=torch.tensor([len(wav)], device=self.device))
        enc = enc[0] if isinstance(enc, tuple) else enc
        lp = self.model.ctc.log_softmax(enc).float()
        normalized = normalize_text(text)
        ids = self.converter.tokens2ids(self.tokenizer.text2tokens(normalized)) if normalized else []
        ids = [i for i in ids if i >= 0 and i != self.blank]
        def nll(tokens):
            if not tokens:
                return lp.new_tensor(0.)
            target = torch.tensor(tokens, device=self.device)[None]
            value = torch.nn.functional.ctc_loss(lp.transpose(0, 1), target, lengths.long(),
                torch.tensor([len(tokens)], device=self.device), blank=self.blank, reduction="sum", zero_infinity=False)
            return value/len(tokens)
        values = ctc_statistics(lp[0, :int(lengths[0])], self.blank, nll(ids), nll(self.prompt+ids))
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite Dolphin CTC statistics; check transcript/audio alignment.")
        return values.cpu()


def prepare_ctc(records, config, output, device):
    from ..utils.common import file_hash, save_json
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"source.json").exists():
        validate_cache(config, output)
    scorer = DolphinStatistics(config, device)
    for i, record in enumerate(records):
        values = scorer(load_audio(record["audio"]), record.get("text", ""))
        torch.save(dict(ctc=values), output/f"{ctc_key(record)}.pt")
        if (i+1) % 100 == 0:
            print(f"Dolphin: {i+1}/{len(records)}", flush=True)
    save_json(output/"source.json", {k:dict(path=str(Path(v).resolve()), sha256=file_hash(v))
                                    for k,v in config.items()})


def validate_cache(config, output):
    import json
    from ..utils.common import file_hash
    recorded = json.loads((Path(output)/"source.json").read_text(encoding="utf-8"))
    current = {k:dict(path=str(Path(v).resolve()), sha256=file_hash(v)) for k,v in config.items()}
    if recorded != current:
        raise ValueError("Dolphin cache belongs to different pretrained files; regenerate it in a new directory.")
