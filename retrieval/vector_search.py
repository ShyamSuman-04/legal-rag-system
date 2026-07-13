"""
vector_search.py

Performs semantic vector search using Elasticsearch KNN
and BAAI/bge-small-en-v1.5 embeddings.

Input
-----
Elasticsearch index (ELASTIC_INDEX_NAME), populated by the
embedding/indexing stage with a 384-dim "embedding" field per chunk.

Output
------
Ranked list of chunk-level search results (see SearchUtils.format_hit
for the exact shape), ready to be merged with keyword_search.py
results inside hybrid_search.py.

Notes
-----
The embedding model can be injected (see `model` param on __init__)
so that HybridSearch - which needs both KeywordSearch and
VectorSearch alive at once - can load SentenceTransformer(...) a
single time and hand the same instance to VectorSearch, instead of
paying the model-load cost again on every VectorSearch() construction.

Author : Shyam Suman
"""

import logging
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

from config import (
    ELASTIC_INDEX_NAME,
    EMBEDDING_MODEL_NAME,
    BGE_QUERY_PREFIX,
)

from retrieval.search_utils import SearchUtils

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# KNN tuning constants
# ---------------------------------------------------------------------
# num_candidates controls how many vectors the HNSW graph walk visits
# per shard before returning the top-k. Higher = better recall, more
# latency. With a corpus in the low thousands of chunks (not tiny),
# a wider candidate pool than the bare Elastic default is worth the
# extra latency for a legal-citation use case where recall matters.

MIN_NUM_CANDIDATES = 200
NUM_CANDIDATES_MULTIPLIER = 20


# ---------------------------------------------------------------------
# Vector Search
# ---------------------------------------------------------------------

class VectorSearch:
    """
    Semantic search over the chunk index using dense vector (KNN)
    retrieval. Embeds the query with BAAI/bge-small-en-v1.5 and asks
    Elasticsearch for the nearest chunks by cosine similarity on the
    "embedding" field.
    """

    def __init__(self, model: Optional[SentenceTransformer] = None):
        """
        Parameters
        ----------
        model : SentenceTransformer, optional
            An already-loaded embedding model to reuse. Pass this in
            from HybridSearch (or any caller holding multiple search
            components at once) so the model is loaded exactly once
            per process instead of once per VectorSearch instance.
            If omitted, VectorSearch loads its own copy.
        """

        self.utils = SearchUtils()

        self.client = self.utils.get_client()

        if model is None:

            logger.info("=" * 65)
            logger.info("Loading embedding model...")
            logger.info("Model : %s", EMBEDDING_MODEL_NAME)

            self.model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device="cpu",
            )

            logger.info("Embedding model loaded successfully.")
            logger.info("=" * 65)

        else:

            logger.info("Reusing shared embedding model instance.")
            self.model = model

    # -----------------------------------------------------------------
    # Query Embedding
    # -----------------------------------------------------------------

    def embed_query(
        self,
        query: str,
    ) -> List[float]:
        """
        Converts a user query into a dense embedding.

        BGE models are trained asymmetrically: passages are embedded
        as-is, but queries need an instruction prefix
        (BGE_QUERY_PREFIX) to retrieve well against them. Never apply
        this prefix on the document/chunk side - only here.
        """

        prefixed_query = BGE_QUERY_PREFIX + query

        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    # -----------------------------------------------------------------
    # KNN Request Builder
    # -----------------------------------------------------------------

    def build_knn_query(
        self,
        query_vector: List[float],
        top_k: int = 10,
        document_type: Optional[str] = None,
        allowed_document_ids: Optional[List[str]] = None,
        allowed_chunk_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Builds an Elasticsearch KNN search request body.

        Optional filters narrow the candidate set before the KNN
        search runs (e.g. a document_type facet, or a graph-derived
        allowlist of document/chunk ids from Neo4j). All filters are
        combined with AND (bool.must).
        """

        knn = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": max(
                MIN_NUM_CANDIDATES,
                top_k * NUM_CANDIDATES_MULTIPLIER,
            ),
        }

        filter_queries = []

        if document_type:

            filter_queries.append(
                {
                    "term": {
                        "document_type": document_type
                    }
                }
            )

        if allowed_document_ids:

            filter_queries.append(
                {
                    "terms": {
                        "document_id": allowed_document_ids
                    }
                }
            )

        if allowed_chunk_ids:

            filter_queries.append(
                {
                    "terms": {
                        "chunk_id": allowed_chunk_ids
                    }
                }
            )

        if filter_queries:

            knn["filter"] = {
                "bool": {
                    "must": filter_queries
                }
            }

        return {
            "knn": knn,
            "size": top_k,
        }

    # -----------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_type: Optional[str] = None,
        allowed_document_ids: Optional[List[str]] = None,
        allowed_chunk_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Executes semantic vector search and returns formatted hits.
        """

        query = self.utils.validate_query(query)

        logger.info("=" * 65)
        logger.info("Running Vector Search")
        logger.info("Query : %s", query)
        logger.info("Top K : %d", top_k)
        logger.info("=" * 65)

        try:

            logger.info("Generating query embedding...")

            query_vector = self.embed_query(query)

            request = self.build_knn_query(
                query_vector=query_vector,
                top_k=top_k,
                document_type=document_type,
                allowed_document_ids=allowed_document_ids,
                allowed_chunk_ids=allowed_chunk_ids,
            )

            response = self.client.search(
                index=ELASTIC_INDEX_NAME,
                **request,
            )

        except Exception as error:

            logger.error("Vector search failed : %s", error)

            # Intentionally re-raised rather than returning [].
            # An empty list here is indistinguishable from "zero
            # relevant chunks", which would hide a real ES/network
            # failure from HybridSearch (and from anyone comparing
            # BM25 vs. vector recall). Callers that want graceful
            # degradation - e.g. HybridSearch falling back to
            # keyword-only results if the vector engine is down -
            # should catch RuntimeError around this call, not have
            # it silently swallowed here.
            raise RuntimeError("Vector search failed.") from error

        hits = response.get(
            "hits", {}
        ).get(
            "hits", []
        )

        logger.info(
            "Retrieved %d semantic results.",
            len(hits),
        )

        results = self.utils.format_hits(hits)

        return results

    # -----------------------------------------------------------------
    # Interactive CLI
    # -----------------------------------------------------------------

    def search_interactive(self):
        """
        Interactive command-line interface for semantic search.
        """
        print("\n")
        print("=" * 80)
        print("US TAX & LEGAL RAG SYSTEM")
        print("Semantic Vector Search")
        print("=" * 80)
        print("Type 'exit' or 'quit' to stop.\n")

        while True:

            try:

                query = input("Enter query: ").strip()

                if query.lower() in {"exit", "quit"}:
                    print("\nExiting Vector Search...\n")
                    break

                if not query:
                    print("Please enter a valid query.\n")
                    continue

                results = self.search(query)

                self.utils.print_results(results)

            except KeyboardInterrupt:

                print("\nInterrupted by user.")
                break

            except Exception as error:

                logger.exception(error)
                print(f"\nError: {error}\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    try:

        vector_search = VectorSearch()
        vector_search.search_interactive()

    except Exception as error:

        logger.exception(error)
        print("\nFailed to start Vector Search.\n")


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()