"""
keyword_search.py

Performs BM25 keyword search on the Elasticsearch index for the
US Tax & Legal RAG System.

Author: Shyam Suman
Project: US Tax & Legal RAG System
"""

import logging
from typing import Dict, List, Optional

from config import ELASTIC_INDEX_NAME

from retrieval.search_utils import SearchUtils

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class KeywordSearch:
    """
    BM25-based keyword search using Elasticsearch.
    """

    def __init__(self):

        self.utils = SearchUtils()

        self.client = self.utils.get_client()

    # -----------------------------------------------------------------

    def build_query(
        self,
        query: str,
        top_k: int = 10,
        document_type: Optional[str] = None,
        allowed_document_ids: Optional[List[str]] = None,
        allowed_chunk_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Builds the Elasticsearch BM25 query.

        Field boosting uses the flattened top-level fields written
        by index_documents.py's build_es_document() — confirmed
        that it flattens metadata.sections -> sections,
        metadata.usc_citations -> usc_citations,
        metadata.entities.acts -> acts, metadata.entities.cases ->
        cases, metadata.keywords -> keywords, etc. Boosting
        "metadata.sections" directly would NOT hit these (that
        path only exists inside the nested, dynamically-mapped
        "metadata" object kept for provenance) — the flattened
        top-level fields are the actual, intended boost targets:

            document_name    -> highest priority (exact doc match)
            sections         -> exact statutory section numbers
            usc_citations    -> exact US Code citations
            cfr_citations    -> exact CFR citations
            acts             -> named statutes
            cases            -> named judgments ("Party v. Party")
            keywords         -> corpus-level TF-IDF keywords
            document_type    -> mild boost (acts/judgments/tax/pov)
            text             -> default weight, free-text fallback

        Optional filters (document_type, allowed_document_ids,
        allowed_chunk_ids) let a future graph_retriever.py (Section
        8.7.1 of the build spec) narrow the candidate set before
        BM25 runs, without changing this method's core query shape.
        """

        multi_match_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    "document_name^4",
                    "sections^3",
                    "usc_citations^3",
                    "cfr_citations^3",
                    "acts^2",
                    "cases^2",
                    "keywords^2",
                    "document_type^1.5",
                    "text",
                ],
                "type": "best_fields",
                "operator": "or",
            }
        }

        filter_clauses = self._build_filter_clauses(
            document_type=document_type,
            allowed_document_ids=allowed_document_ids,
            allowed_chunk_ids=allowed_chunk_ids,
        )

        if filter_clauses:
            es_query = {
                "bool": {
                    "must": multi_match_query,
                    "filter": filter_clauses,
                }
            }
        else:
            es_query = multi_match_query

        return {
            "size": top_k,
            "query": es_query,
            "highlight": {
                "fields": {
                    "text": {
                        "fragment_size": 200,
                        "number_of_fragments": 2,
                    }
                }
            },
        }

    def _build_filter_clauses(
        self,
        document_type: Optional[str],
        allowed_document_ids: Optional[List[str]],
        allowed_chunk_ids: Optional[List[str]],
    ) -> List[Dict]:
        """
        Translates optional narrowing parameters into Elasticsearch
        filter clauses. Filters do not affect BM25 scoring, only
        which documents are eligible to be scored at all — this is
        the exact hook a future graph_retriever.py plugs into
        (Neo4j -> allowed document_ids -> filtered BM25 search).
        """

        filter_clauses: List[Dict] = []

        if document_type:
            filter_clauses.append(
                {"term": {"document_type": document_type}}
            )

        if allowed_document_ids:
            filter_clauses.append(
                {"terms": {"document_id": allowed_document_ids}}
            )

        if allowed_chunk_ids:
            filter_clauses.append(
                {"terms": {"chunk_id": allowed_chunk_ids}}
            )

        return filter_clauses

    # -----------------------------------------------------------------

    def _attach_highlights(
        self,
        results: List[Dict],
        hits: List[Dict],
    ) -> List[Dict]:
        """
        SearchUtils.format_hit() (the shared, canonical formatter
        used by keyword/vector/hybrid search alike) intentionally
        doesn't know about highlighting, since only keyword search
        requests it. Attaching it here, post-formatting, keeps
        search_utils.py generic while still surfacing highlighted
        fragments for a nicer CLI/UI result.

        results and hits are the same length and same order (both
        derived from response["hits"]["hits"]), so a positional zip
        is safe.
        """

        for result, hit in zip(results, hits):

            highlight_fragments = hit.get("highlight", {}).get("text", [])

            result["highlight"] = highlight_fragments

        return results

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
        Executes BM25 keyword search and returns formatted,
        citation-ready results, each carrying a "highlight" list of
        matched text fragments alongside the standard
        SearchUtils.format_hit() fields.
        """

        query = self.utils.validate_query(query)

        logger.info("=" * 65)
        logger.info("Running Keyword Search")
        logger.info("Query : %s", query)
        logger.info("Top K : %d", top_k)

        if document_type:
            logger.info("Filter: document_type = %s", document_type)

        if allowed_document_ids:
            logger.info(
                "Filter: %d allowed document_id(s)",
                len(allowed_document_ids),
            )

        if allowed_chunk_ids:
            logger.info(
                "Filter: %d allowed chunk_id(s)",
                len(allowed_chunk_ids),
            )

        logger.info("=" * 65)

        body = self.build_query(
            query=query,
            top_k=top_k,
            document_type=document_type,
            allowed_document_ids=allowed_document_ids,
            allowed_chunk_ids=allowed_chunk_ids,
        )

        # Pass query/highlight/size as explicit keyword args rather
        # than body=body — the Elasticsearch 8.x/9.x Python client
        # is moving away from the single "body" blob in favor of
        # top-level params matching the request shape. body's keys
        # ("size", "query", "highlight") already match those kwarg
        # names, so this is a plain, safe unpack.
        response = self.client.search(
            index=ELASTIC_INDEX_NAME,
            **body,
        )

        hits = response.get("hits", {}).get("hits", [])

        logger.info("Retrieved %d results.", len(hits))

        results = self.utils.format_hits(hits)

        results = self._attach_highlights(results, hits)

        return results

    # -----------------------------------------------------------------

    def search_interactive(self):
        """
        Interactive command-line search loop.
        """

        print("\n")
        print("=" * 80)
        print("US TAX & LEGAL RAG SYSTEM")
        print("Keyword Search (BM25)")
        print("=" * 80)
        print("Type 'exit' or 'quit' to stop.\n")

        while True:

            try:
                query = input("Enter query: ").strip()

                if query.lower() in {"exit", "quit"}:
                    print("\nExiting Keyword Search...\n")
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
        keyword_search = KeywordSearch()

        keyword_search.search_interactive()

    except Exception as error:
        logger.exception(error)
        print("\nFailed to start Keyword Search.\n")


if __name__ == "__main__":
    main()