from pymilvus import connections, db, exceptions

DATABASE_NAME = "beat_maker"
COLLECTION_NAME = "beat_maker"

try:
    connections.connect(
        host="localhost",
        port="19530",
        db_name="default",
    )
except exceptions.MilvusException as e:
    print("Error: {}".format(e))


db.drop_database(DATABASE_NAME)

print("Available databases:", db.list_database())
