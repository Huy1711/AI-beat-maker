from pymilvus import MilvusClient, connections, db, exceptions, utility

DATABASE_NAME = "beat_maker"
COLLECTION_NAME = "beat_maker"
INDEX_NAME = "vector_index"
MILVUS_URL = "http://localhost:19530"

try:
    connections.connect(
        host="localhost",
        port="19530",
        db_name="default",
    )
except exceptions.MilvusException as e:
    print("Error: {}".format(e))

db.using_database(DATABASE_NAME)

client = MilvusClient(uri=MILVUS_URL, db_name=DATABASE_NAME)
index_params = MilvusClient.prepare_index_params()

client.drop_collection(collection_name=COLLECTION_NAME)

print("Available collections:", utility.list_collections())
