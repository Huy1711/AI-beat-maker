from pymilvus import MilvusClient

DATABASE_NAME = "beat_maker"
COLLECTION_NAME = "beat_maker"
INDEX_NAME = "vector_index"
MILVUS_URL = "http://localhost:19530"

client = MilvusClient(uri=MILVUS_URL, db_name=DATABASE_NAME)

client.drop_index(collection_name=COLLECTION_NAME, index_name=INDEX_NAME)

res = client.list_indexes(collection_name=COLLECTION_NAME)
print("Available indices:", res)
