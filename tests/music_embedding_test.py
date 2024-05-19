import torchaudio
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

SAMPLE_RATE = 8000
SEGMENT_SIZE = int(SAMPLE_RATE * 1.0)
HOP_SIZE = int(SAMPLE_RATE * 0.5)
EXAMPLE_DATA = "samples/001001.wav"

TRITON_SERVER_URL = "localhost:8001"  # Triton server ip:grpc_port

transformation = torchaudio.transforms.MelSpectrogram(
    sample_rate=8000,
    n_fft=1024,
    hop_length=256,
    n_mels=256,
    f_min=300,
    f_max=4000,
)


def prepare_input_feature(example_file):
    wav, sr = torchaudio.load(example_file)
    segments = wav.squeeze().unfold(0, SEGMENT_SIZE, HOP_SIZE)
    segments = segments - segments.mean(dim=1).unsqueeze(1)
    features = transformation(segments)
    features = features.clamp(1e-5).log()
    features = features.numpy()
    return features, sr


def service_smoke_test():
    Audio, _ = prepare_input_feature(EXAMPLE_DATA)
    print("input Audio", Audio.shape)

    inputs = [
        grpcclient.InferInput(
            "input",
            Audio.shape,
            np_to_triton_dtype(Audio.dtype),
        ),
    ]
    inputs[0].set_data_from_numpy(Audio)

    outputs = [
        grpcclient.InferRequestedOutput("output"),
    ]
    with grpcclient.InferenceServerClient(url=TRITON_SERVER_URL) as triton_client:
        response = triton_client.infer(
            "neuralfp", inputs, request_id=str(1), outputs=outputs
        )

        # result = response.get_response()
        output_data = response.as_numpy("output")
    print("output_data", output_data.shape)
    return output_data


if __name__ == "__main__":
    service_smoke_test()
