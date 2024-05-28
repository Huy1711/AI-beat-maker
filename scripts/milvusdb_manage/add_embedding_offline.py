"""This script extract audio files in the folder EXTRACT_FOLDER into embeddings
and add to Milvus Vector DB. The extract and adding can be faster using multiprocessing.
"""

import glob
import os

import numpy as np
import torch
import torchaudio
import tqdm
from pymilvus import MilvusClient

DATABASE_NAME = "beat_maker"
COLLECTION_NAME = "beat_maker"
INDEX_NAME = "vector_index"
MILVUS_URL = "http://localhost:19530"
SAVE_MODEL_PATH = "./deploy/music_embedding/model_repository/neuralfp/1/model.pt"
MILVUS_ADD_CHUNK_SIZE = 10_000
SAMPLE_RATE = 8_000
EXTRACT_FOLDER = "./datasets/neural-audio-fp-dataset/music/test-query-db-500-30s/db/"

file_list = glob.glob(os.path.join(EXTRACT_FOLDER, "**/*.wav"))
device = "cpu"
loaded_model = torch.jit.load(SAVE_MODEL_PATH, map_location=device)
loaded_model.eval()
milvus_client = MilvusClient(uri=MILVUS_URL, db_name=DATABASE_NAME)

transformation = torchaudio.transforms.MelSpectrogram(
    sample_rate=8000,
    n_fft=1024,
    hop_length=256,
    n_mels=256,
    f_min=300,
    f_max=4000,
)


def realpath_to_id(path):
    if isinstance(path, int):
        path = str(path)
    return os.path.basename(os.path.splitext(path)[0])


def split_to_equal_chunk(arr: np.array, chunk_size):
    arr = np.array_split(arr, np.ceil(len(arr) / chunk_size))
    return arr


def prepare_feature(file, segment_size=8000, hop_size=4000):
    """Load audio, resample if needed, and extract MelSpectrogram feature"""
    audio_format = file.split(".")[-1]  ## currently support wav, mp3
    wav, sr = torchaudio.load(file, format=audio_format)
    if sr != SAMPLE_RATE:
        transform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        wav = transform(wav)
    wav = wav.mean(dim=0)
    ## slice wav into segments
    segments = wav.unfold(0, segment_size, hop_size)
    segments = segments - segments.mean(dim=1).unsqueeze(1)
    ## extract mel-spectrogram
    features = transformation(segments)
    features = features.clamp(1e-5).log()
    return features, sr


insert_segments_total_count = 0
for filepath in tqdm.tqdm(file_list, desc="Adding audio embeddings to DB"):
    feature, sr = prepare_feature(filepath)
    feature = feature.to(device)

    embeddings = loaded_model(feature)

    data = []
    file_id = realpath_to_id(filepath)
    for offset, embedding in enumerate(embeddings):
        data.append(
            {
                "file_id": file_id,
                "offset": offset,
                "embedding": embedding,
            }
        )

    chunked_data_list = split_to_equal_chunk(data, chunk_size=MILVUS_ADD_CHUNK_SIZE)
    for data_chunk in chunked_data_list:
        res = milvus_client.insert(
            collection_name=COLLECTION_NAME,
            data=data_chunk.tolist(),
        )
    insert_segments_total_count += len(data)

print(
    f"Totally added {insert_segments_total_count} segments to the collection {COLLECTION_NAME} of database {DATABASE_NAME}"
)
