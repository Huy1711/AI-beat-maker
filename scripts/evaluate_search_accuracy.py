"""
This script evaluate accuracy of audio fingerprint model by utilizing the
deployed Milvus Vector DB and neuralFP model using Triton Server.
Please turn those services on before running this scripts.
"""

import glob
import sys
from collections import Counter

import tqdm

sys.path.insert(0, "./")
from deploy.api.utils.search.music_database_client import MusicDatabaseClient
from deploy.api.utils.search.music_embedding_client import MusicEmbeddingClient

TRITON_SERVER_URL = "localhost:8001"
MILVUS_SERVER_URL = "http://localhost:19530"

query_list = glob.glob(
    "/home/huynd/Code/AI-beat-maker/datasets/neural-audio-fp-dataset/music/test-query-db-500-30s/query_fixed_SNR/snr_0dB_1s/**/*.wav"
)

Triton_helper = MusicEmbeddingClient(TRITON_SERVER_URL)
Milvus_helper = MusicDatabaseClient(MILVUS_SERVER_URL)

if __name__ == "__main__":
    results = []
    for filepath in tqdm.tqdm(query_list[:3]):
        Embeddings = Triton_helper.get_embeddings(filepath)
        Search_Results = Milvus_helper.search_embeddings(Embeddings)
        results.append((filepath, Search_Results))


Search_Results = [query_res[1] for query_res in results]
True_Results = [query_res[0].split("/")[-1][:-4] for query_res in results]

Final_Results = []
for result in Search_Results:
    guessed_file_ids = Counter([segment[0]["entity"]["file_id"] for segment in result])
    major_voting_result = guessed_file_ids.most_common(1)[0][0]
    Final_Results.append(major_voting_result)

assert len(Final_Results) == len(True_Results)

count = 0
for pred, label in zip(Final_Results, True_Results):
    if pred == label:
        count += 1

print("Accuracy:", count / len(Final_Results))
