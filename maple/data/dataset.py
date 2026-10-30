import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from .cache import ctc_key

def load_audio(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if rate != 16000:
        from math import gcd
        divisor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000//divisor, rate//divisor).astype(np.float32)
    if not len(audio):
        raise ValueError(f"Empty audio: {path}")
    return audio

def load_video(path):
    import cv2
    # Input is the mouth ROI sequence, NOT an unlocalized full face.
    reader = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    reader.release()
    if not frames:
        raise ValueError(f"No video frames decoded: {path}")
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()/255
    gray = (video*torch.tensor([.2989, .5870, .1140])[None, :, None, None]).sum(1, keepdim=True)
    gray = torch.nn.functional.interpolate(gray, (88, 88), mode="bilinear", align_corners=False)
    return (gray-.421)/.165

class RawDataset(Dataset):
    def __init__(self, records, modality, ctc_dir=None):
        self.records, self.modality = records, modality
        self.ctc_dir = Path(ctc_dir) if ctc_dir else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        signal = load_audio(record["audio"]) if self.modality == "audio" else load_video(record["video"])
        ctc = None
        if self.ctc_dir is not None:
            key = ctc_key(record)
            ctc = torch.load(self.ctc_dir / f"{key}.pt", map_location="cpu", weights_only=True)["ctc"]
        return record, signal, ctc


def collate_raw(batch):
    records, signals, ctc = zip(*batch)
    return list(records), list(signals), torch.stack(ctc) if ctc[0] is not None else None


def raw_loader(records, modality, ctc_dir, train=False, batch_size=None, workers=4):
    size = (16 if train or modality == "audio" else 8) if batch_size is None else batch_size
    return DataLoader(RawDataset(records, modality, ctc_dir), batch_size=size,
                      shuffle=train, num_workers=workers, collate_fn=collate_raw)
