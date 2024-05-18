import torch
import torch.nn as nn

from .encoder import SepConvEncoder
from .projector import LinearProjector


class NeuralAudioFingerprinter(nn.Module):
    def __init__(
        self,
        d: int = 128,
        h: int = 1024,
        u: int = 32,
        in_F: int = 256,
        sample_rate: int = 8000,
        segment_size: float = 1.0,
        stft_hop: int = 256,
    ):
        super(NeuralAudioFingerprinter, self).__init__()

        self.encoder = SepConvEncoder(
            d,
            h,
            in_F,
            segment_size,
            stft_hop,
            sample_rate,
        )
        self.projector = LinearProjector(d, h, u)

    def forward(self, xs, norm=True) -> torch.Tensor:
        xs = self.encoder(xs)
        out = self.projector(xs, norm=norm)
        return out