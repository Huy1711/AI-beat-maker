from pymilvus import connections, db, exceptions, utility

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

res = utility.index_building_progress(
    collection_name=COLLECTION_NAME, index_name=INDEX_NAME
)

print(f"Index {INDEX_NAME} of collection {COLLECTION_NAME} info:", res)
