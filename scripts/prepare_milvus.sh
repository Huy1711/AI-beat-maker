#!/bin/bash

python ./scripts/milvusdb_manage/create_database.py
python ./scripts/milvusdb_manage/create_collection.py
python ./scripts/milvusdb_manage/add_embedding_offline.py
python ./scripts/milvusdb_manage/create_index.py
