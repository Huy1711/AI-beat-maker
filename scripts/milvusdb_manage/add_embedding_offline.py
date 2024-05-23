import glob

import torch
import torchaudio
from pymilvus import MilvusClient

from deploy.api.utils.common import realpath_to_id, split_to_equal_chunk

DATABASE_NAME = "beat-maker"
COLLECTION_NAME = "beat-maker"
INDEX_NAME = "vector_index"
MILVUS_URL = "http://localhost:19530"
SAVE_MODEL_PATH = "./deploy/music_embedding/model_repository/neuralfp/1/model.pt"
MILVUS_ADD_CHUNK_SIZE = 10_000

file_list = glob.glob("")
device = "cpu"
loaded_model = torch.jit.load(SAVE_MODEL_PATH, map_location=device)
loaded_model.eval()
milvus_client = MilvusClient(uri=MILVUS_URL, db_name=DATABASE_NAME)

insert_total_count = 0
for filepath in file_list:
    wav, sr = torchaudio.load(filepath)
    wav = wav.squeeze()
    wav = wav.to(device)

    embeddings = loaded_model(wav)

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
            collection_name="audiofp",
            data=data_chunk.tolist(),
        )
    insert_total_count += len(data)
