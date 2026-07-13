"""
reranker.py

CrossEncoder-based reranking for the US Tax & Legal RAG System.

Pipeline
--------
User Query
      │
      ▼
Hybrid Search (BM25 + Vector + RRF)
      │
Top 20 retrieved chunks
      ▼
CrossEncoder
      │
Top 8 reranked chunks
      ▼
LLM

Author: Shyam Suman
Project: US Tax & Legal RAG System
"""

import logging
from typing import Dict, List, Optional
import torch

from sentence_transformers import CrossEncoder

from config import (
    RERANKER_MODEL_NAME,
    RERANK_TOP_K,
)

from retrieval.hybrid_search import HybridSearch

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------


class Reranker:
    """
    CrossEncoder reranker.

    Hybrid search retrieves candidate chunks.

    CrossEncoder then performs pairwise semantic comparison
    between

        (Query, Chunk)

    and assigns a true relevance score.

    Unlike embedding similarity, CrossEncoder jointly
    encodes the query and document, making it significantly
    more accurate.
    """

    def __init__(
        self,
        hybrid_search: Optional[HybridSearch] = None,
        model: Optional[CrossEncoder] = None,
    ):

        logger.info("=" * 65)
        logger.info("Initializing CrossEncoder Reranker")

        # -----------------------------------------
        # Hybrid Search
        # -----------------------------------------

        self.hybrid_search = (
            hybrid_search
            if hybrid_search is not None
            else HybridSearch()
        )

        # -----------------------------------------
        # CrossEncoder
        # -----------------------------------------

        if model is None:

            logger.info(
                "Loading CrossEncoder model..."
            )

            logger.info(
                "Model : %s",
                RERANKER_MODEL_NAME,
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.model = CrossEncoder(
                RERANKER_MODEL_NAME,
                device=device,
            )

        else:

            self.model = model

        logger.info(
            "CrossEncoder loaded successfully."
        )

        logger.info("=" * 65)

    # -----------------------------------------------------------------
    # Core Reranking Logic
    # -----------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = RERANK_TOP_K,
    ) -> List[Dict]:
        """
        Reranks Hybrid Search results.

        Parameters
        ----------
        query:
            User question.

        results:
            Output of HybridSearch.search()

        top_k:
            Number of chunks to return.

        Returns
        -------
        List[Dict]

            Same chunk dictionaries, now enriched with

                rerank_score

            sorted by descending semantic relevance.
        """

        if not results:

            return []

        logger.info("=" * 65)
        logger.info("Running CrossEncoder Reranker")
        logger.info("Candidates : %d", len(results))
        logger.info("=" * 65)

        # -----------------------------------------
        # Build (query, chunk) pairs
        # -----------------------------------------

        sentence_pairs = [

            (
                query,
                chunk["text"],
            )

            for chunk in results

        ]

        # -----------------------------------------
        # CrossEncoder prediction
        # -----------------------------------------

        logger.info(
            "Scoring query-document pairs..."
        )

        scores = self.model.predict(
            sentence_pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        # -----------------------------------------
        # Attach scores
        # -----------------------------------------

        enriched_results = []

        for chunk, score in zip(results, scores):

            chunk = dict(chunk)

            chunk["rerank_score"] = float(score)

            enriched_results.append(chunk)

        # -----------------------------------------
        # Sort
        # -----------------------------------------

        enriched_results.sort(

            key=lambda x: x["rerank_score"],

            reverse=True,

        )

        logger.info(
            "Top reranked score : %.4f",
            enriched_results[0]["rerank_score"],
        )

        return enriched_results[:top_k]

    # -----------------------------------------------------------------
    # Search Pipeline
    # -----------------------------------------------------------------

    def search(
        self,
        query: str,
        candidate_pool_size: int = 50,
        top_k: int = RERANK_TOP_K,
        document_type: Optional[str] = None,
        allowed_document_ids: Optional[List[str]] = None,
        allowed_chunk_ids: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Complete retrieval pipeline.

        Query
            ↓
        Hybrid Search
            ↓
        Top candidate_pool_size chunks
            ↓
        CrossEncoder
            ↓
        Top top_k chunks

        Parameters
        ----------
        query
            User question.

        candidate_pool_size
            Number of chunks retrieved by Hybrid Search
            before reranking.

        top_k
            Number of chunks returned after reranking.

        document_type
            Optional filter.

        allowed_document_ids
            Optional GraphRAG filter.

        allowed_chunk_ids
            Optional GraphRAG filter.
        """

        logger.info("=" * 65)
        logger.info("Running Retrieval + Reranking Pipeline")
        logger.info("Query : %s", query)
        logger.info("Candidate Pool : %d", candidate_pool_size)
        logger.info("Final Top K : %d", top_k)
        logger.info("=" * 65)

        # -------------------------------------------------------------
        # Hybrid Retrieval
        # -------------------------------------------------------------

        hybrid_results = self.hybrid_search.search(
            query=query,
            top_k=candidate_pool_size,
            candidate_pool_size=candidate_pool_size,
            document_type=document_type,
            allowed_document_ids=allowed_document_ids,
            allowed_chunk_ids=allowed_chunk_ids,
        )

        logger.info(
            "Hybrid Search returned %d chunks.",
            len(hybrid_results),
        )

        # -------------------------------------------------------------
        # CrossEncoder Reranking
        # -------------------------------------------------------------

        reranked_results = self.rerank(
            query=query,
            results=hybrid_results,
            top_k=top_k,
        )

        logger.info(
            "Returning %d reranked chunks.",
            len(reranked_results),
        )

        return reranked_results

    # -----------------------------------------------------------------
    # Pretty Printing
    # -----------------------------------------------------------------

    @staticmethod
    def print_results(
        results: List[Dict],
    ):
        """
        Prints reranked search results.
        """

        print("\n")
        print("=" * 90)
        print("RERANKED RESULTS")
        print("=" * 90)

        if not results:

            print("No documents found.")
            print("=" * 90)
            return

        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")
            print("-" * 90)

            print(
                "Document      :",
                result["document_name"],
            )

            print(
                "Type          :",
                result["document_type"],
            )

            print(
                "Pages         :",
                f'{result["page_start"]}-{result["page_end"]}',
            )

            print(
                "Matched By    :",
                result.get("matched_by", "-"),
            )

            print(
                "Hybrid Score  :",
                round(
                    result.get(
                        "hybrid_score",
                        0,
                    ),
                    6,
                ),
            )

            print(
                "Rerank Score  :",
                round(
                    result["rerank_score"],
                    6,
                ),
            )

            print()

            snippet = result["text"][:500]

            snippet = snippet.replace(
                "\n",
                " ",
            )

            print(snippet)

            if result.get("highlight"):

                print()

                print("Highlights:")

                for fragment in result["highlight"]:

                    fragment = fragment.replace(
                        "\n",
                        " ",
                    )

                    print(
                        "...",
                        fragment,
                        "...",
                    )

            print()

    # -----------------------------------------------------------------
    # Interactive CLI
    # -----------------------------------------------------------------

    def search_interactive(self):
        """
        Interactive command-line interface.

        Runs:

            Query
                ↓
            Hybrid Search
                ↓
            CrossEncoder Reranker
                ↓
            Display Results
        """

        print("\n")
        print("=" * 90)
        print("US TAX & LEGAL RAG SYSTEM")
        print("CrossEncoder Reranker")
        print("=" * 90)
        print("Type 'exit' or 'quit' to stop.\n")

        while True:

            try:

                query = input("Enter query: ").strip()

                if query.lower() in {"exit", "quit"}:

                    print("\nExiting Reranker...\n")
                    break

                if not query:

                    print("Please enter a valid query.\n")
                    continue

                results = self.search(query)

                self.print_results(results)

            except KeyboardInterrupt:

                print("\nInterrupted by user.")
                break

            except Exception as error:

                logger.exception(error)

                print(
                    f"\nError: {error}\n"
                )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    """
    Entry point.

    Example
    -------
    python -m retrieval.reranker
    """

    try:

        reranker = Reranker()

        reranker.search_interactive()

    except Exception as error:

        logger.exception(error)

        print(
            "\nFailed to start Reranker.\n"
        )


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()