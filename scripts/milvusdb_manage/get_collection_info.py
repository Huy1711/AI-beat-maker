from pymilvus import MilvusClient

DATABASE_NAME = "beat_maker"
COLLECTION_NAME = "beat_maker"
MILVUS_URL = "http://localhost:19530"

client = MilvusClient(uri=MILVUS_URL, db_name=DATABASE_NAME)

res = client.get_collection_stats(collection_name=COLLECTION_NAME)

print(f"Collection {COLLECTION_NAME} stats:", res)
