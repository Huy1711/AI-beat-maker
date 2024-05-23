query_list = glob.glob(
    "/data/huynd49/dataset/music/fma_medium/FMA_queries/snr-6_queries/*.wav"
)

query_list__JOBs = hanoi_utils.split_list_evenly(
    Inp_List=query_list,
    max_splits=4,
)


def compute_embedding(query_list):
    Triton_helper = TritonHelper(TRITON_SERVER_URL)
    Milvus_helper = MilvusHelper(MILVUS_SERVER_URL)

    for filepath in query_list:
        Embeddings = Triton_helper.get_embeddings(filepath)
        Search_Results = Milvus_helper.search_embeddings(Embeddings, nprobe=50, topk=10)
        yield filepath, Search_Results


results = hanoi_utils.MultiprocsRunner(
    Partial_Functions=[
        functools.partial(
            compute_embedding,
            query_list=query_list__JOB,
        )
        for query_list__JOB in query_list__JOBs
    ],
    print_progress=True,
    total=len(query_list),
    desc="Extract embeddings",
).run()

Search_Results = [query_res[1] for query_res in results]
True_Results = [query_res[0].split("_")[4] for query_res in results]
Final_Results = []
for result in Search_Results:
    Reranked_Results = reranking(
        result,
        segment_chunk_size=10,
        duplicate_segment_thresh=0.0,
        hop_size=0.5,
    )
    Final_Results.append(Reranked_Results)

Final_Results = [pred[0][0] for pred in Final_Results]

assert len(Final_Results) == len(True_Results)

count = 0
for pred, label in zip(Final_Results, True_Results):
    if pred == label:
        count += 1

print("Accuracy:", count / len(Final_Results))
