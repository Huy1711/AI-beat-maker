"""
Audio data augmentation inspired by
https://github.com/stdio2016/pfann/blob/main/datautil/ir.py
https://github.com/stdio2016/pfann/blob/main/datautil/noise.py
"""

import os

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import tqdm

from ..utils.common import load_dataset


class ImpulseResponseNoise(object):
    """Randomly select reverb noise and add to audio"""

    def __init__(
        self,
        dataset_dir: str,
        list_file: str,
        length: float = 1.0,
        segment_size: float = 1.0,
        sample_rate: int = 8000,
    ):
        self.segment_size = int(segment_size * sample_rate)
        noises = load_dataset(list_file)
        ir_len = int(length * sample_rate)
        fft_needed = self.segment_size + length
        self.fftconv_n = 1024
        while self.fftconv_n <= fft_needed:
            self.fftconv_n *= 2
        self.noises = []
        for noise in tqdm.tqdm(noises, desc="Loading IR"):
            filename = os.path.join(dataset_dir, noise["audio_filepath"])
            smp, sr = torchaudio.load(filename)
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
    """Randomly select background noise and add to audio
    (e.g. speech, laughter, vehicle sounds, etc.)
    """

    def __init__(
        self,
        dataset_dir: str,
        list_file: str,
        cache_dir: str,
        snr_min: int = 0,
        snr_max: int = 10,
        sample_rate: int = 8000,
    ):
        self.snr_min = snr_min
        self.snr_max = snr_max
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_name = os.path.basename(list_file).split(".")[-2]
        self.cache_file = os.path.join(cache_dir, cache_file_name + "_bg.npy")

        self.data = self.load_from_cache()
        if self.data is None:
            noises = load_dataset(list_file)
            save_file = open(self.cache_file, "ab")
            for noise in tqdm.tqdm(noises, desc="Loading Background noise"):
                filename = os.path.join(dataset_dir, noise["audio_filepath"])
                smp, sr = torchaudio.load(filename)
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
    """
    Randomly cut original audio into fixed size segment
        (e.g. randomly clip 1s from 1.2s audio with
        segment_offset = 1.2 * sample_rate; segment_size = 1 * sample_rate)
    """

    def __init__(
        self,
        segment_offset: float = 1.2,
        segment_size: float = 1.0,
        sample_rate: int = 8000,
    ):
        self.sample_rate = sample_rate
        self.segment_offset = int(segment_offset * self.sample_rate)
        self.segment_size = int(segment_size * self.sample_rate)

    def apply(self, xs: torch.Tensor) -> torch.Tensor:
        offset_range = self.segment_offset - self.segment_size
        rand_offsets = torch.randint(high=offset_range, size=(len(xs),)).tolist()
        xs_aug = [
            xi[offset : offset + min(self.segment_size, xi.shape[-1])]
            for xi, offset in zip(xs, rand_offsets)
        ]
        xs_aug = torch.stack(xs_aug)
        return xs_aug
