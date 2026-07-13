"""
search_utils.py

Common utility functions for retrieval modules.

Used by:
    - keyword_search.py
    - vector_search.py
    - hybrid_search.py

Author: Shyam Suman
"""

import logging
from typing import Dict, List

from indexing.elastic_client import ElasticClient

# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------


class SearchUtils:

    def __init__(self):

        self.client = ElasticClient().get_client()

    # -----------------------------------------------------------------

    def get_client(self):
        """
        Returns Elasticsearch client.
        """
        return self.client

    # -----------------------------------------------------------------

    @staticmethod
    def validate_query(query: str):

        if query is None:
            raise ValueError("Query cannot be None.")

        query = query.strip()

        if len(query) == 0:
            raise ValueError("Query cannot be empty.")

        return query

    # -----------------------------------------------------------------

    @staticmethod
    def format_hit(hit: Dict):

        source = hit["_source"]

        return {

            "chunk_id":
                source.get("chunk_id"),

            "document_id":
                source.get("document_id"),

            "document_name":
                source.get("document_name"),

            "document_type":
                source.get("document_type"),

            "source_file":
                source.get("source_file"),

            "page_start":
                source.get("page_start"),

            "page_end":
                source.get("page_end"),

            "score":
                round(hit["_score"], 4),

            "text":
                source.get("text"),

            "metadata":
                source.get("metadata", {})

        }

    # -----------------------------------------------------------------

    def format_hits(
        self,
        hits: List[Dict]
    ) -> List[Dict]:

        return [
            self.format_hit(hit)
            for hit in hits
        ]

    # -----------------------------------------------------------------

    @staticmethod
    def print_results(results: List[Dict]):

        print("\n")

        print("=" * 80)

        print("SEARCH RESULTS")

        print("=" * 80)

        if not results:

            print("No documents found.")

            print("=" * 80)

            return

        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")

            print("-" * 80)

            print("Document :", result["document_name"])

            print("Type     :", result["document_type"])

            print(
                "Pages    :",
                f'{result["page_start"]}-{result["page_end"]}'
            )

            print("Score    :", result["score"])

            print()

            snippet = result["text"][:350]

            snippet = snippet.replace("\n", " ")

            print(snippet)

            print()

        print("=" * 80)