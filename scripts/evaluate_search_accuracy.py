import glob
import sys

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
    for filepath in tqdm.tqdm(query_list):
        Embeddings = Triton_helper.get_embeddings(filepath)
        Search_Results = Milvus_helper.search_embeddings(Embeddings)
        results.append((filepath, Search_Results))

print(results[0])


# Search_Results = [query_res[1] for query_res in results]
# True_Results = [query_res[0].split("_")[4] for query_res in results]
# Final_Results = []
# for result in Search_Results:
#     Reranked_Results = reranking(
#         result,
#         segment_chunk_size=10,
#         duplicate_segment_thresh=0.0,
#         hop_size=0.5,
#     )
#     Final_Results.append(Reranked_Results)

# Final_Results = [pred[0][0] for pred in Final_Results]

# assert len(Final_Results) == len(True_Results)

# count = 0
# for pred, label in zip(Final_Results, True_Results):
#     if pred == label:
#         count += 1

# print("Accuracy:", count / len(Final_Results))
