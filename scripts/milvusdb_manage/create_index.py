from pymilvus import MilvusClient

DATABASE_NAME = "beat_maker"
COLLECTION_NAME = "beat_maker"
INDEX_NAME = "vector_index"
MILVUS_URL = "http://localhost:19530"

client = MilvusClient(uri=MILVUS_URL, db_name=DATABASE_NAME)
index_params = MilvusClient.prepare_index_params()

# Add an index on the vector field.
index_params.add_index(
    field_name="embedding",
    metric_type="IP",
    index_type="IVF_PQ",  # Change to "GPU_IVF_PQ" if using GPU
    params={"nlist": 200, "m": 16, "nbits": 8},
    index_name=INDEX_NAME,
)

# Create an index file
client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)

res = client.list_indexes(collection_name=COLLECTION_NAME)

print("Available indices:", res)
