"""
elastic_client.py

Creates and manages the Elasticsearch client.

Author: Divyansh Kumar
Project: US Tax & Legal RAG System
"""

import logging

from elasticsearch import Elasticsearch

from config import (
    ELASTIC_API_KEY,
    ELASTIC_CLOUD_ID,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class ElasticClient:
    """
    Creates a reusable Elasticsearch client.
    """

    def __init__(self):

        if not ELASTIC_CLOUD_ID:
            raise ValueError(
                "ELASTIC_CLOUD_ID not found in environment."
            )

        if not ELASTIC_API_KEY:
            raise ValueError(
                "ELASTIC_API_KEY not found in environment."
            )

        logger.info("=" * 60)
        logger.info("Connecting to Elasticsearch...")

        try:

            self.client = Elasticsearch(
                cloud_id=ELASTIC_CLOUD_ID,
                api_key=ELASTIC_API_KEY,
                request_timeout=30,
            )

            if not self.client.ping():
                raise ConnectionError(
                    "Unable to connect to Elasticsearch."
                )

            logger.info("Connected successfully.")

        except Exception as e:

            logger.exception("Elasticsearch connection failed.")

            raise RuntimeError(
                "Could not establish Elasticsearch connection."
            ) from e

        logger.info("=" * 60)

    # -------------------------------------------------------------

    def get_client(self):
        """
        Returns the Elasticsearch client.
        """
        return self.client


# ---------------------------------------------------------------------
# Local Test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    elastic = ElasticClient()

    client = elastic.get_client()

    info = client.info()

    print("\nConnected!\n")

    print("Cluster Name :", info["cluster_name"])

    print("Version :", info["version"]["number"])