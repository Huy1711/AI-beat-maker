import os
from typing import List, Optional

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import tqdm

from ..utils.common import load_dataset


class IR(object):
    def __init__(self, list_file, length, segment_size, sample_rate=8000):
        self.segment_size = segment_size * sample_rate
        noises = load_dataset(list_file)
        ir_len = int(length * sample_rate)
        fft_needed = self.segment_size + length
        self.fftconv_n = 1024
        while self.fftconv_n <= fft_needed:
            self.fftconv_n *= 2
        self.noises = []
        for noise in tqdm.tqdm(noises, desc="Loading IR"):
            smp, sr = torchaudio.load(noise["audio_filepath"])
            if sr != sample_rate:
                raise Exception(f"IR's sample rate is {sr}Hz not {sample_rate}Hz")
            smp = smp.squeeze()
            truncated = smp[:ir_len]
            freqd = torch.fft.rfft(truncated, self.fftconv_n)
            self.noises.append(freqd)
        self.noises = torch.stack(self.noises)

    def random_choose(self, num):
        indices = torch.randint(0, self.noises.shape[0], size=(num,), dtype=torch.long)
        return self.noises[indices]

    def apply(self, xs: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.rfft(xs, self.fftconv_n)
        ir = self.random_choose(spec.shape[0])
        spec *= ir
        xs_aug = torch.fft.irfft(spec, self.fftconv_n)
        xs_aug = xs_aug[..., : self.segment_size]
        return xs_aug


class BackGroundNoise(object):
    def __init__(self, list_file, snr_min, snr_max, cache_dir, sample_rate=8000):
        self.snr_min = snr_min
        self.snr_max = snr_max
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_name = os.path.basename(list_file).split(".")[-2]
        self.cache_file = os.path.join(cache_dir, cache_file_name + ".npy")

        self.data = self.load_from_cache()
        if self.data is None:
            noises = load_dataset(list_file)
            save_file = open(self.cache_file, "ab")
            for noise in tqdm.tqdm(noises, desc="Loading Background noise"):
                name = noise["audio_filepath"]
                smp, sr = torchaudio.load(name)
                if sr != sample_rate:
                    raise Exception(
                        f"Background noise's sample rate is {sr}Hz not {sample_rate}Hz"
                    )
                save_file.write(smp.numpy().tobytes())
                save_file.flush()
            save_file.close()
            self.data = self.load_from_cache()

    def load_from_cache(self):
        if os.path.exists(self.cache_file):
            return np.memmap(self.cache_file, dtype=np.float32, mode="r")
        return None

    def random_choose(self, num, duration):
        indices = torch.randint(
            0, self.data.shape[0] - duration, size=(num,), dtype=torch.long
        )
        out = torch.zeros([num, duration], dtype=torch.float32)
        for i in range(num):
            start = int(indices[i])
            noise = self.data[start : start + duration].copy()
            out[i] = torch.from_numpy(noise)
        return out

    def apply(self, xs: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        noise = self.random_choose(xs.shape[0], xs.shape[1])
        vol_xs = torch.clamp((xs.square()).mean(dim=1), min=eps).sqrt()
        vol_noise = torch.clamp((noise.square()).mean(dim=1), min=eps).sqrt()
        snr = torch.FloatTensor(xs.shape[0]).uniform_(self.snr_min, self.snr_max)
        ratio = vol_xs / vol_noise
        ratio *= 10 ** -(snr / 20)
        xs_aug = xs + ratio.unsqueeze(1) * noise
        return xs_aug


class RandomClip(object):
    def __init__(self, sample_rate, segment_offset, segment_size):
        self.sample_rate = sample_rate
        self.segment_offset = int(segment_offset * self.sample_rate)
        self.segment_size = int(segment_size * self.sample_rate)

    def apply(self, xs: torch.Tensor) -> torch.Tensor:
        offset_range = int(self.segment_offset - self.segment_size)
        rand_offsets = torch.randint(high=offset_range, size=(len(xs),)).tolist()
        xs_aug = [
            xi[offset : offset + min(self.segment_size, xi.shape[-1])]
            for xi, offset in zip(xs, rand_offsets)
        ]
        xs_aug = torch.stack(xs_aug)
        return xs_aug