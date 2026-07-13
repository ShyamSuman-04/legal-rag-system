"""
hybrid_search.py

Fuses BM25 keyword search (keyword_search.py) and dense vector
search (vector_search.py) into a single ranked list using
Reciprocal Rank Fusion (RRF).

Author: Shyam Suman
Project: US Tax & Legal RAG System

Pipeline position
------------------
    User Query
        |
        v
    Hybrid Search (this file)   <- BM25 + Vector, fused with RRF
        |
    Top 20 ranked chunks
        |
        v
    Cross Encoder Reranker      <- next file to build
        |
    Top 10 chunks
        |
        v
    Context Builder -> LLM

Why RRF instead of averaging scores
-----------------------------------
BM25 scores are unbounded (a term-frequency-weighted score that can be
0.5 or 25 depending on the corpus), and cosine similarity from the
vector KNN is bounded in [-1, 1] (in practice usually 0-1 after
normalization). Averaging or summing these two directly would let
whichever engine happens to produce larger numbers dominate the
fusion, for reasons that have nothing to do with relevance. RRF
sidesteps this by throwing away the raw scores and fusing purely on
*rank position* in each engine's own result list:

    score(chunk) = sum over every ranked list it appears in of
                   1 / (k + rank_in_that_list)

A chunk that shows up near the top of both BM25 and vector results
rises to the top of the fused list; a chunk that only one engine
found still gets a (smaller) score instead of being dropped, which is
what keeps hybrid search more forgiving than either engine alone.

Graceful degradation
--------------------
If one engine throws (e.g. Elasticsearch hiccups on the KNN clause
but BM25 still works, or vice versa), hybrid_search logs a warning
and fuses on whatever the surviving engine returned rather than
failing the whole request. Only if *both* engines fail does this
module raise - see search() below.
"""

import logging
from typing import Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer

from retrieval.keyword_search import KeywordSearch
from retrieval.vector_search import VectorSearch

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Fusion tuning constants
# ---------------------------------------------------------------------
# RRF_K: the standard smoothing constant from Cormack et al. (2009).
#   Higher k flattens the curve (rank 1 vs rank 10 matters less);
#   lower k rewards top ranks more aggressively. 60 is the
#   widely-used default and a reasonable starting point here.
#
# CANDIDATE_POOL_SIZE: how many raw hits each engine contributes
#   *before* fusion. This needs to be comfortably larger than the
#   final HYBRID_TOP_K, or RRF has nothing to fuse - if both engines
#   only returned 20 hits each, fusion just re-sorts the same 20-ish
#   chunks instead of surfacing chunks that one engine ranked highly
#   but the other almost missed. 50 matches the "top 50 candidates
#   feed the reranker" step in the wider retrieval pipeline.
#
# HYBRID_TOP_K: the default size of the final fused list this module
#   returns. The reranker is free to ask for more (e.g.
#   hybrid_search.search(query, top_k=50)) if it wants a wider net to
#   rerank than this module's own default.

RRF_K = 60
CANDIDATE_POOL_SIZE = 50
HYBRID_TOP_K = 20


# ---------------------------------------------------------------------
# Standalone RRF (dependency-free, independently testable)
# ---------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_id_lists: List[List[str]],
    k: int = RRF_K,
) -> Dict[str, float]:
    """
    Pure Reciprocal Rank Fusion over any number of ranked ID lists.

    Parameters
    ----------
    ranked_id_lists : list of list of str
        One ranked list of IDs per engine, best result first.
    k : int
        RRF smoothing constant.

    Returns
    -------
    Dict[str, float]
        id -> fused score. Not sorted - callers sort as needed.
    """

    scores: Dict[str, float] = {}

    for ranked_ids in ranked_id_lists:

        for rank, item_id in enumerate(ranked_ids, start=1):

            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    return scores


# ---------------------------------------------------------------------
# Hybrid Search
# ---------------------------------------------------------------------

class HybridSearch:
    """
    Combines KeywordSearch (BM25) and VectorSearch (dense KNN) via
    Reciprocal Rank Fusion into a single ranked list of chunks.
    """

    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        """
        Parameters
        ----------
        embedding_model : SentenceTransformer, optional
            Pre-loaded embedding model to hand to VectorSearch, so a
            caller that needs to construct multiple search
            components (e.g. an evaluation harness running many
            HybridSearch-like objects) doesn't pay the model-load
            cost more than once. If omitted, VectorSearch loads its
            own copy exactly as it would standalone.
        """

        logger.info("Initializing Hybrid Search...")

        self.keyword_search = KeywordSearch()

        self.vector_search = VectorSearch(model=embedding_model)

        # Reuse the Elasticsearch client/SearchUtils instance already
        # held by keyword_search rather than opening a third
        # ElasticClient connection just to call validate_query().
        self.utils = self.keyword_search.utils

        logger.info("Hybrid Search ready (BM25 + Vector, RRF fusion).")

    # -----------------------------------------------------------------
    # Fusion internals
    # -----------------------------------------------------------------

    @staticmethod
    def _rank_and_score_lookup(
        results: List[Dict],
    ) -> Dict[str, Tuple[int, Optional[float]]]:
        """
        Builds {chunk_id: (rank, engine_score)} from one engine's
        already-ranked result list (rank 1 = best).

        Uses .get("score") rather than ["score"]: RRF itself only
        ever consumes rank order (see reciprocal_rank_fusion), so the
        raw per-engine score is display-only here. If a future change
        to the shared SearchUtils.format_hit() ever drops or renames
        "score", this degrades to None (printed as "-") instead of
        taking down hybrid search with a KeyError.
        """

        lookup: Dict[str, Tuple[int, Optional[float]]] = {}

        for rank, result in enumerate(results, start=1):

            lookup[result["chunk_id"]] = (rank, result.get("score"))

        return lookup

    def _fuse(
        self,
        keyword_results: List[Dict],
        vector_results: List[Dict],
        k: int,
    ) -> List[Dict]:
        """
        Fuses keyword and vector result lists into one ranked list of
        rich chunk records, each annotated with per-engine rank/score
        alongside the combined hybrid_score.
        """

        keyword_lookup = self._rank_and_score_lookup(keyword_results)
        vector_lookup = self._rank_and_score_lookup(vector_results)

        ranked_id_lists = [
            list(keyword_lookup.keys()),
            list(vector_lookup.keys()),
        ]

        hybrid_scores = reciprocal_rank_fusion(ranked_id_lists, k=k)

        # chunk_id -> full formatted record, preferring the keyword
        # version when a chunk was found by both engines (it carries
        # the "highlight" fragments that vector search doesn't
        # produce; everything else in the record is identical, since
        # both go through the same SearchUtils.format_hit()).
        record_lookup: Dict[str, Dict] = {}

        for result in vector_results:
            record_lookup[result["chunk_id"]] = result

        for result in keyword_results:
            record_lookup[result["chunk_id"]] = result

        fused_records: List[Dict] = []

        for chunk_id, hybrid_score in hybrid_scores.items():

            record = dict(record_lookup[chunk_id])

            keyword_rank, keyword_score = keyword_lookup.get(
                chunk_id, (None, None)
            )
            vector_rank, vector_score = vector_lookup.get(
                chunk_id, (None, None)
            )

            if keyword_rank is not None and vector_rank is not None:
                matched_by = "both"
            elif keyword_rank is not None:
                matched_by = "keyword_only"
            else:
                matched_by = "vector_only"

            # "score" always means "the number to sort this record
            # by" regardless of pipeline stage - downstream modules
            # (reranker, context_builder) can rely on it without
            # caring whether it came from BM25, KNN, or RRF fusion.
            record["score"] = round(hybrid_score, 6)
            record["hybrid_score"] = round(hybrid_score, 6)
            record["keyword_rank"] = keyword_rank
            record["keyword_score"] = keyword_score
            record["vector_rank"] = vector_rank
            record["vector_score"] = vector_score
            record["matched_by"] = matched_by
            record.setdefault("highlight", [])

            fused_records.append(record)

        fused_records.sort(key=lambda r: r["hybrid_score"], reverse=True)

        return fused_records

    # -----------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = HYBRID_TOP_K,
        candidate_pool_size: int = CANDIDATE_POOL_SIZE,
        document_type: Optional[str] = None,
        allowed_document_ids: Optional[List[str]] = None,
        allowed_chunk_ids: Optional[List[str]] = None,
        rrf_k: int = RRF_K,
    ) -> List[Dict]:
        """
        Runs BM25 and vector search in parallel (candidate_pool_size
        hits each), fuses them with RRF, and returns the top_k fused
        chunks.

        document_type / allowed_document_ids / allowed_chunk_ids pass
        straight through to both engines unchanged - this is the same
        filter contract keyword_search.py and vector_search.py already
        expose, so a future graph_retriever.py can narrow both engines
        at once without hybrid_search.py needing to change.
        """

        query = self.utils.validate_query(query)

        logger.info("=" * 65)
        logger.info("Running Hybrid Search (BM25 + Vector, RRF)")
        logger.info("Query               : %s", query)
        logger.info("Candidate pool/engine: %d", candidate_pool_size)
        logger.info("Final top_k         : %d", top_k)
        logger.info("=" * 65)

        keyword_results: List[Dict] = []
        vector_results: List[Dict] = []

        keyword_error: Optional[Exception] = None
        vector_error: Optional[Exception] = None

        try:

            keyword_results = self.keyword_search.search(
                query=query,
                top_k=candidate_pool_size,
                document_type=document_type,
                allowed_document_ids=allowed_document_ids,
                allowed_chunk_ids=allowed_chunk_ids,
            )

        except Exception as error:

            keyword_error = error
            logger.warning(
                "Keyword search failed, continuing with "
                "vector-only results : %s",
                error,
            )

        try:

            vector_results = self.vector_search.search(
                query=query,
                top_k=candidate_pool_size,
                document_type=document_type,
                allowed_document_ids=allowed_document_ids,
                allowed_chunk_ids=allowed_chunk_ids,
            )

        except Exception as error:

            vector_error = error
            logger.warning(
                "Vector search failed, continuing with "
                "keyword-only results : %s",
                error,
            )

        if keyword_error is not None and vector_error is not None:

            logger.error(
                "Both engines failed - keyword: %s | vector: %s",
                keyword_error,
                vector_error,
            )

            raise RuntimeError(
                "Hybrid search failed: both keyword and vector "
                "search engines raised errors."
            ) from vector_error

        fused_results = self._fuse(
            keyword_results=keyword_results,
            vector_results=vector_results,
            k=rrf_k,
        )

        both_count = sum(
            1 for r in fused_results if r["matched_by"] == "both"
        )
        keyword_only_count = sum(
            1 for r in fused_results if r["matched_by"] == "keyword_only"
        )
        vector_only_count = sum(
            1 for r in fused_results if r["matched_by"] == "vector_only"
        )

        logger.info(
            "Fusion produced %d unique chunks "
            "(both=%d, keyword_only=%d, vector_only=%d).",
            len(fused_results),
            both_count,
            keyword_only_count,
            vector_only_count,
        )

        top_results = fused_results[:top_k]

        logger.info("Returning top %d hybrid results.", len(top_results))

        return top_results

    # -----------------------------------------------------------------
    # Result Printing (hybrid-specific: BM25 / Vector / Hybrid scores)
    # -----------------------------------------------------------------

    @staticmethod
    def print_results(results: List[Dict]):
        """
        Hybrid-specific results printer.

        SearchUtils.print_results() (shared with keyword_search.py
        and vector_search.py) only knows about a single generic
        "score" field, which is the right amount of detail for either
        engine alone. Hybrid results carry three distinct scores
        (BM25, vector, fused) plus which engine(s) found each chunk,
        so this prints its own richer view instead of overloading the
        shared formatter.
        """

        print("\n")
        print("=" * 80)
        print("HYBRID SEARCH RESULTS (RRF-fused)")
        print("=" * 80)

        if not results:
            print("No documents found.")
            print("=" * 80)
            return

        for i, result in enumerate(results, start=1):

            print(f"\nResult {i}")
            print("-" * 80)
            print("Document      :", result["document_name"])
            print("Type          :", result["document_type"])
            print(
                "Pages         :",
                f'{result["page_start"]}-{result["page_end"]}',
            )
            print("Matched By    :", result["matched_by"])
            print("Hybrid Score  :", result["hybrid_score"])
            print(
                "  BM25 Score  :",
                result["keyword_score"]
                if result["keyword_score"] is not None
                else "-",
                f'(rank {result["keyword_rank"]})'
                if result["keyword_rank"] is not None
                else "",
            )
            print(
                "  Vector Score:",
                result["vector_score"]
                if result["vector_score"] is not None
                else "-",
                f'(rank {result["vector_rank"]})'
                if result["vector_rank"] is not None
                else "",
            )
            print()

            snippet = result["text"][:350]
            snippet = snippet.replace("\n", " ")
            print(snippet)

            if result.get("highlight"):
                print()
                print("Highlights:")
                for fragment in result["highlight"]:
                    print("  ...", fragment.replace("\n", " "), "...")

            print()

        print("=" * 80)

    # -----------------------------------------------------------------
    # Interactive CLI
    # -----------------------------------------------------------------

    def search_interactive(self):
        """
        Interactive command-line search loop.
        """

        print("\n")
        print("=" * 80)
        print("US TAX & LEGAL RAG SYSTEM")
        print("Hybrid Search (BM25 + Vector, RRF Fusion)")
        print("=" * 80)
        print("Type 'exit' or 'quit' to stop.\n")

        while True:

            try:
                query = input("Enter query: ").strip()

                if query.lower() in {"exit", "quit"}:
                    print("\nExiting Hybrid Search...\n")
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
                print(f"\nError: {error}\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():

    try:
        hybrid_search = HybridSearch()

        hybrid_search.search_interactive()

    except Exception as error:
        logger.exception(error)
        print("\nFailed to start Hybrid Search.\n")


if __name__ == "__main__":
    main()