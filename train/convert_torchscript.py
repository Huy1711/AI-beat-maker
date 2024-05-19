import torch
import torchaudio
from omegaconf import OmegaConf

from neuralfp.model.neuralfp import NeuralAudioFingerprinter

SAMPLE_RATE = 8000
SEGMENT_SIZE = int(SAMPLE_RATE * 1.0)
HOP_SIZE = int(SAMPLE_RATE * 0.5)
EXAMPLE_DATA = "../datasets/neural-audio-fp-dataset/music/test-dummy-db-100k-full/from_fma_large10k-30s_for_mini/000/000138.wav"
CONFIG_PATH = "configs/train.yaml"
MODEL_CHECKPOINT = "artifacts/neuralfp_epoch88.pt"
MODEL_SAVE_PATH = "artifacts/model.pt"

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
    return features, sr

def load_torch_model():
    config = OmegaConf.load(CONFIG_PATH)
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location="cpu")
    model = NeuralAudioFingerprinter(**config["model"]["neuralfp"])
    model.load_state_dict(checkpoint["state_dict"]["model"])
    model.eval()
    return model


def convert_model():

    model = load_torch_model()

    example = torch.rand(19, 256, 32)

    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(MODEL_SAVE_PATH)
    print(f"Torchscript model has been successfully saved to {MODEL_SAVE_PATH}")


def test_output():
    
    jit_model = torch.jit.load(MODEL_SAVE_PATH)
    
    torch_model = load_torch_model()

    example_data, _ = prepare_input_feature(EXAMPLE_DATA)

    jit_output = jit_model(example_data)
    torch_output = torch_model(example_data)

    assert torch.allclose(jit_output, torch_output), \
        "Pytorch model output and TorchScript model output are not the same"


if __name__ == "__main__":
    convert_model()
    test_output()
