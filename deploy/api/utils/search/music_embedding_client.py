import logging
import time

import numpy as np
import torchaudio
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

from ..common import split_to_equal_chunk

EXTRACT_EMB_CHUNK_SIZE = 1024
SAMPLE_RATE = 8000
SEGMENT_SIZE = 1.0
HOP_SIZE = 0.5

logger = logging.getLogger("beat-maker-api")
logging.basicConfig(level=logging.INFO)


class MusicEmbeddingClient(object):
    def __init__(self, url):
        self.url = url
        self.transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=256,
            n_mels=256,
            f_min=300,
            f_max=4000,
        )
        self.segment_size = int(SEGMENT_SIZE * SAMPLE_RATE)
        self.hop_size = int(HOP_SIZE * SAMPLE_RATE)
        self.model_name = "neuralfp"

    def prepare_feature(self, file):
        audio_format = file.filename.split(".")[-1]  ## currently support wav, mp3
        wav, sr = torchaudio.load(file.file, format=audio_format)
        if sr != SAMPLE_RATE:
            transform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            wav = transform(wav)
        wav = wav.mean(dim=0)
        ## slice wav into segments
        segments = wav.unfold(0, self.segment_size, self.hop_size)
        segments = segments - segments.mean(dim=1).unsqueeze(1)
        ## extract mel-spectrogram
        features = self.transformation(segments)
        features = features.clamp(1e-5).log()
        features = features.numpy()
        return features, sr

    def get_embeddings(self, file) -> np.ndarray:
        start = time.time()
        Audio, _ = self.prepare_feature(file)

        Audio_Segments = split_to_equal_chunk(Audio, chunk_size=EXTRACT_EMB_CHUNK_SIZE)

        ## Send embedding requests to Triton server
        Output_Embeddings = []
        for input_audio in Audio_Segments:
            inputs = [
                grpcclient.InferInput(
                    "input",
                    input_audio.shape,
                    np_to_triton_dtype(input_audio.dtype),
                ),
            ]
            inputs[0].set_data_from_numpy(input_audio)

            outputs = [
                grpcclient.InferRequestedOutput("output"),
            ]
            with grpcclient.InferenceServerClient(url=self.url) as triton_client:
                response = triton_client.infer(
                    self.model_name, inputs, request_id=str(1), outputs=outputs
                )

            # result = response.get_response()
            output_data = response.as_numpy("output")
            Output_Embeddings.append(output_data)
        Output_Embeddings = np.concatenate(Output_Embeddings)
        logger.info(f"Time extract embeddings: {time.time() - start}")
        return Output_Embeddings
