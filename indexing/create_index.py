"""
create_index.py

Creates the Elasticsearch index for the
US Tax & Legal RAG System.

Features
--------
✓ Checks whether the index already exists
✓ Asks before deleting an existing index
✓ Creates an optimized mapping
✓ Supports dense vector search
✓ Preserves all metadata required for legal citations

Author: Divyansh Kumar
"""

import logging

from config import (
    ELASTIC_INDEX_NAME,
    EMBEDDING_DIM,
)

from indexing.elastic_client import ElasticClient

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class IndexCreator:

    def __init__(self):
        self.client = ElasticClient().get_client()

    # -----------------------------------------------------------------

    def delete_existing_index(self):

        if not self.client.indices.exists(index=ELASTIC_INDEX_NAME):
            return

        print("\nIndex already exists.")

        choice = input(
            "Delete and recreate the index? (y/n): "
        ).strip().lower()

        if choice != "y":
            logger.info("Keeping existing index.")
            return False

        logger.info("Deleting existing index...")

        self.client.indices.delete(
            index=ELASTIC_INDEX_NAME
        )

        logger.info("Existing index deleted.")

        return True

    # -----------------------------------------------------------------

    def create_index(self):

        if self.client.indices.exists(index=ELASTIC_INDEX_NAME):

            deleted = self.delete_existing_index()

            if deleted is False:
                return

        logger.info(
            "Creating index '%s'...",
            ELASTIC_INDEX_NAME
        )

        mapping = {

            "settings": {

                "number_of_shards": 1,
                "number_of_replicas": 0

            },

            "mappings": {

                "properties": {

                    # -------------------------------------------------
                    # Document Information
                    # -------------------------------------------------

                    "chunk_id": {
                        "type": "keyword"
                    },

                    "document_id": {
                        "type": "keyword"
                    },

                    "document_name": {
                        "type": "keyword"
                    },

                    "document_type": {
                        "type": "keyword"
                    },

                    "source_file": {
                        "type": "keyword"
                    },

                    # -------------------------------------------------
                    # Chunk Information
                    # -------------------------------------------------

                    "chunk_index": {
                        "type": "integer"
                    },

                    "page_start": {
                        "type": "integer"
                    },

                    "page_end": {
                        "type": "integer"
                    },

                    "page_ids": {
                        "type": "keyword"
                    },

                    "character_count": {
                        "type": "integer"
                    },

                    "word_count": {
                        "type": "integer"
                    },

                    "chunked_at": {
                        "type": "date"
                    },

                    # -------------------------------------------------
                    # Searchable Text
                    # -------------------------------------------------

                    "text": {
                        "type": "text"
                    },

                    # -------------------------------------------------
                    # Metadata
                    # -------------------------------------------------

                    "metadata": {
                        "type": "object",
                        "dynamic": True
                    },

                    # -------------------------------------------------
                    # Dense Vector
                    # -------------------------------------------------

                    "embedding": {

                        "type": "dense_vector",

                        "dims": EMBEDDING_DIM,

                        "index": True,

                        "similarity": "cosine"

                    }

                }

            }

        }

        self.client.indices.create(
            index=ELASTIC_INDEX_NAME,
            body=mapping
        )

        logger.info("Index created successfully.")

    # -----------------------------------------------------------------

    def print_summary(self):

        mapping = self.client.indices.get_mapping(
            index=ELASTIC_INDEX_NAME
        )

        fields = mapping[
            ELASTIC_INDEX_NAME
        ]["mappings"]["properties"]

        logger.info("=" * 65)
        logger.info("Index Summary")
        logger.info("=" * 65)

        logger.info("Index Name : %s", ELASTIC_INDEX_NAME)

        logger.info("Embedding Dimension : %d", EMBEDDING_DIM)

        logger.info("Total Fields : %d", len(fields))

        logger.info("-" * 65)

        for field, value in fields.items():

            logger.info(
                "%-20s %s",
                field,
                value["type"]
            )

        logger.info("=" * 65)

    # -----------------------------------------------------------------

    def run(self):

        logger.info("=" * 65)
        logger.info("Creating Elasticsearch Index")
        logger.info("=" * 65)

        self.create_index()

        self.print_summary()


# ---------------------------------------------------------------------


def main():

    creator = IndexCreator()

    creator.run()


if __name__ == "__main__":
    main()