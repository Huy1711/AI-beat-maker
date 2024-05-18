import os
import typing as tp

import torch
import torch.nn.functional as F
import torchaudio
from omegaconf import DictConfig
from torch.utils.data import Dataset

from ..utils.common import load_dataset
from .augment import BackGroundNoise, ImpulseResponseNoise, RandomClip


def collate_data(batch: tp.List[torch.Tensor]) -> tp.Tuple[torch.Tensor, ...]:
    features = [b[0] for b in batch]
    features = torch.cat(features)
    targets = [b[1] for b in batch]
    targets = torch.cat(targets)
    return features, targets


class MusicSegmentDataset(Dataset):
    """
    Data preparation by cutting music files into segments
    The implementation is inspired by
        https://github.com/stdio2016/pfann/blob/main/datautil/dataset_v2.py
    The difference in this implementation is the __getitem__() function
        aims for file-level batching instead of segment-level batching
    """

    def __init__(self, config: DictConfig):
        self.dataset_dir = config["dataset_dir"]
        self.dataset = load_dataset(config["audio_list_file"])
        self.sample_rate = config["sample_rate"]
        self.segment_offset = int(config["segment_offset"] * self.sample_rate)
        self.segment_size = int(config["segment_size"] * self.sample_rate)
        self.hop_size = int(config["hop_size"] * self.sample_rate)

        self.random_clip = RandomClip(
            segment_offset=config["segment_offset"],
            segment_size=config["segment_size"],
            sample_rate=config["sample_rate"],
        )

        self.augments = []
        if config.get("augmentation"):
            self.augments = [
                BackGroundNoise(**config["augmentation"]["background"]),
                ImpulseResponseNoise(**config["augmentation"]["ir"]),
            ]

        self.transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=256,
            f_min=300,
            f_max=4000,
        )

    def _extract_feature(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.transformation(waveform)
        features = features.clamp(1e-5).log()
        return features

    def _cut_audio_to_segments(self, audio) -> tp.List[torch.Tensor]:
        """
        Cut audio to segments with offset
        """
        audio_duration = audio.shape[1]
        num_segs = (audio_duration - self.segment_size + self.hop_size) // self.hop_size
        segments = []
        for idx in range(num_segs):
            segment_start = idx * self.hop_size
            segment_duration = min(self.segment_offset, audio_duration - segment_start)
            segment = audio[:, segment_start : segment_start + segment_duration]
            if segment_duration < self.segment_offset:
                segment = F.pad(segment, (0, self.segment_offset - segment_duration))
            segments.append(segment)
        segments = torch.stack(segments)
        return segments

    def __getitem__(self, idx: int) -> tp.Tuple[torch.Tensor, ...]:
        data = self.dataset[idx]
        audio, sr = torchaudio.load(
            os.path.join(self.dataset_dir, data["audio_filepath"])
        )  # [1, T]

        if sr != self.sample_rate:
            raise Exception(
                f"Training audio's sample rate is {sr}Hz not {self.sample_rate}Hz"
            )

        segments = self._cut_audio_to_segments(audio)
        segments = segments.squeeze(1)  # [N, T'=1.2s]
        targets = segments.clone().detach()

        segments = self.random_clip.apply(segments)  # [N, T'=1s]
        targets = self.random_clip.apply(targets)

        for augment in self.augments:
            segments = augment.apply(segments)  # [N, T'=1s]

        features = self._extract_feature(segments)  # [N, n_mels, T'=1s]
        targets = self._extract_feature(targets)

        return features, targets

    def __len__(self):
        return len(self.dataset)
