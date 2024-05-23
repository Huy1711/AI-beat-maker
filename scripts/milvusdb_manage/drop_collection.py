from pymilvus import MilvusClient, utility

DATABASE_NAME = "beat-maker"
COLLECTION_NAME = "beat-maker"
INDEX_NAME = "vector_index"
MILVUS_URL = "http://localhost:19530"

client = MilvusClient(uri=MILVUS_URL)
index_params = MilvusClient.prepare_index_params()

client.drop_collection(collection_name=COLLECTION_NAME)

print("Available collections:", utility.list_collections())
