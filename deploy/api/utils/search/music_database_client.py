import logging
import time

from pymilvus import MilvusClient

from ..common import split_to_equal_chunk

# Refer to https://milvus.io/docs/limitations.md#Search-limits
MILVUS_SEARCH_SIZE_LIMIT = 16_384
MILVUS_ADD_CHUNK_SIZE = 10_000


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class MusicDatabaseClient(object):
    """
    This class provide some helper functions to interact
    with Milvus Vector Database for the music search feature
    """

    def __init__(self, url):
        self.url = url
        self.db_name = "beat_maker"
        self.collection_name = "beat_maker"
        ## Connect to Milvus server
        self.milvus_client = MilvusClient(uri=url, db_name=self.db_name)

    def search_embeddings(self, Embeddings):
        """
        Split the query embeddings if its size larger than
        the search limit and send those splited search requests.
        """
        start = time.time()
        self.load_collection()

        Embeddings = split_to_equal_chunk(
            Embeddings, chunk_size=MILVUS_SEARCH_SIZE_LIMIT
        )

        Search_Results = []
        for embedding in Embeddings:
            Search_Results += self._milvus_search_embeddings(embedding)

        logger.info(f"Time search embeddings: {time.time() - start}")
        return Search_Results

    def _milvus_search_embeddings(self, embeddings, nprobe=50, topk=1):
        search_params = {
            "metric_type": "IP",
            "params": {
                "nprobe": nprobe,
            },
        }

        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=embeddings,
            anns_field="embedding",
            search_params=search_params,
            limit=topk,  # number of results to return
            output_fields=[
                "file_id",
                "offset",
            ],  # the names of the fields to retrieve with.
            consistency_level="Bounded",
        )
        return results

    def load_collection(self):
        self.milvus_client.load_collection(collection_name=self.collection_name)

    def close(self):
        self.milvus_client.close()
