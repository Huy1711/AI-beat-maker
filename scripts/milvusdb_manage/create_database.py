from pymilvus import connections, db, exceptions

DATABASE_NAME = "beat-maker"
COLLECTION_NAME = "beat-maker"

try:
    connections.connect(
        host="localhost",
        port="19530",
        db_name="default",
    )
except exceptions.MilvusException as e:
    print("Error: {}".format(e))


database = db.create_database(DATABASE_NAME)

print("Available databases:", db.list_database())
