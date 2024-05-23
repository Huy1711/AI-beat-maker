from pymilvus import MilvusClient

DATABASE_NAME = "beat-maker"
COLLECTION_NAME = "beat-maker"
INDEX_NAME = "vector_index"
MILVUS_URL = "http://localhost:19530"

client = MilvusClient(uri=MILVUS_URL)

client.drop_index(collection_name=COLLECTION_NAME, index_name=INDEX_NAME)

res = client.list_indexes(collection_name=COLLECTION_NAME)
print("Available indices:", res)
