from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    db,
    exceptions,
    utility,
)

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


db.using_database(DATABASE_NAME)

id_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True,
    description="primary id",
)

file_id_field = FieldSchema(
    name="file_id",
    dtype=DataType.VARCHAR,
    max_length=256,
    # is_partition_key=True,
    description="file id",
)

offset_field = FieldSchema(
    name="offset",
    dtype=DataType.INT32,
    description="embedding offset from start of file",
)

embedding_field = FieldSchema(
    name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128, description="vector"
)

schema = CollectionSchema(
    fields=[id_field, file_id_field, offset_field, embedding_field],
    enable_dynamic_field=False,
    description="audiofp collection",
)

collection = Collection(
    name=COLLECTION_NAME, schema=schema, using="default", shards_num=2  # Milvus server
)

print("Available collections:", utility.list_collections())
