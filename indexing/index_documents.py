"""
index_documents.py

Bulk-loads chunks_with_embeddings.json into the Elasticsearch
index created by create_index.py, for the US Tax & Legal RAG
System.

Pipeline
--------
chunks_with_embeddings.json
        |
        v
Read JSON  (N chunk dicts)
        |
        v
Validate + transform each chunk into an ES document
  (flatten a few metadata.* fields to top level for BM25
   field-boosting; keep the full metadata object too)
        |
        v
helpers.bulk()  -> one HTTP round trip per batch, not per doc
        |
        v
Refresh index + verify document count matches what was indexed

Why Bulk API and not one-by-one inserts?
-----------------------------------------
Indexing ~7,000+ documents one HTTP call at a time is slow and
hammers the cluster with round-trip overhead. The Bulk API sends
many documents in a single HTTP request; this is the standard,
production way to load data into Elasticsearch and is roughly
20-50x faster for this volume.

Author: Divyansh Kumar
"""

import json
import logging
from typing import Dict, Iterator, List, Optional, Tuple

from elasticsearch import helpers

from config import (
    ELASTIC_INDEX_NAME,
    EMBEDDING_DIM,
    CHUNKS_WITH_EMBEDDINGS_JSON,
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

# ---------------------------------------------------------------------
# Bulk indexing batch size.
# 500 is a reasonable default for ~400-word chunks + a 384-dim
# vector each; tune down if the Elastic Cloud trial tier rejects
# large bulk payloads.
# ---------------------------------------------------------------------

BULK_CHUNK_SIZE = 500


class DocumentIndexer:

    def __init__(self):

        self.client = ElasticClient().get_client()

        self.total_chunks = 0

        self.skipped_chunks = 0

        self.indexed_success = 0

        self.indexed_errors: List[Dict] = []

    # -----------------------------------------------------------------

    def load_chunks(self) -> List[Dict]:

        logger.info(
            "Loading %s", CHUNKS_WITH_EMBEDDINGS_JSON
        )

        with open(
            CHUNKS_WITH_EMBEDDINGS_JSON,
            "r",
            encoding="utf-8",
        ) as file:

            chunks = json.load(file)

        self.total_chunks = len(chunks)

        logger.info(
            "%d chunks loaded.", self.total_chunks
        )

        return chunks

    # -----------------------------------------------------------------

    def build_es_document(self, chunk: Dict) -> Optional[Dict]:
        """
        Convert one chunk_with_embedding dict into the flat
        document shape the Elasticsearch index expects.

        Returns None (and logs a warning) if the chunk is missing
        required fields or has a malformed embedding, so one bad
        chunk never aborts the whole bulk load.
        """

        chunk_id = chunk.get("chunk_id")

        embedding = chunk.get("embedding")

        if not chunk_id:
            logger.warning("Skipping chunk with no chunk_id.")
            return None

        if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIM:

            logger.warning(
                "Skipping chunk '%s': embedding missing or not "
                "%d-dimensional.",
                chunk_id,
                EMBEDDING_DIM,
            )
            return None

        metadata = chunk.get("metadata", {}) or {}

        entities = metadata.get("entities", {}) or {}

        document = {
            "chunk_id": chunk_id,
            "document_id": chunk.get("document_id"),
            "document_name": chunk.get("document_name"),
            "document_type": chunk.get("document_type"),
            "source_file": chunk.get("source_file"),
            "chunk_index": chunk.get("chunk_index"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "page_ids": chunk.get("page_ids", []),
            "character_count": chunk.get("character_count"),
            "word_count": chunk.get("word_count"),
            "chunked_at": chunk.get("chunked_at"),
            "text": chunk.get("text", ""),

            # Full metadata object preserved for provenance /
            # future fields, kept as a dynamic nested object per
            # create_index.py's mapping.
            "metadata": metadata,

            # -----------------------------------------------------
            # Flattened fields for BM25 field-boosting at query
            # time (multi_match on sections^3, usc_citations^3,
            # acts^2, etc. per the hybrid search design). These are
            # duplicates of nested metadata fields, promoted to
            # top level because Elasticsearch cannot field-boost
            # into a dynamic object efficiently.
            # -----------------------------------------------------
            "keywords": metadata.get("keywords", []),
            "sections": metadata.get("sections", []),
            "usc_citations": metadata.get("usc_citations", []),
            "cfr_citations": metadata.get("cfr_citations", []),
            "articles": metadata.get("articles", []),
            "chapters": metadata.get("chapters", []),
            "titles": metadata.get("titles", []),
            "tax_forms": metadata.get("tax_forms", []),
            "irs_publications": metadata.get("irs_publications", []),
            "tax_years": metadata.get("tax_years", []),
            "acts": entities.get("acts", []),
            "code_references": entities.get("code_references", []),
            "cases": entities.get("cases", []),
            "courts": entities.get("courts", []),
            "judges": entities.get("judges", []),
            "author": entities.get("author"),

            "embedding": embedding,
        }

        return self._drop_empty_fields(document)

    def _drop_empty_fields(self, document: Dict) -> Dict:
        """
        Drop None / empty-list / empty-string fields so the
        flattened, mostly-empty per-document-type fields (e.g.
        "courts" on an Acts chunk) don't clutter every document.
        Elasticsearch handles missing fields natively, so this is
        purely for readability/index size, not correctness.
        """

        return {
            key: value
            for key, value in document.items()
            if not (
                value is None
                or (isinstance(value, (list, str)) and len(value) == 0)
            )
        }

    # -----------------------------------------------------------------

    def generate_bulk_actions(
        self,
        chunks: List[Dict],
    ) -> Iterator[Dict]:
        """
        Yield one Elasticsearch bulk "index" action per valid
        chunk. Using a generator (rather than building a full list
        in memory) keeps memory flat even for large corpora, since
        helpers.bulk() consumes actions lazily in batches of
        BULK_CHUNK_SIZE.
        """

        for chunk in chunks:

            es_document = self.build_es_document(chunk)

            if es_document is None:
                self.skipped_chunks += 1
                continue

            yield {
                "_index": ELASTIC_INDEX_NAME,
                "_id": es_document["chunk_id"],
                "_source": es_document,
            }

    # -----------------------------------------------------------------

    def run_bulk_index(self, chunks: List[Dict]) -> Tuple[int, List[Dict]]:
        """
        Stream all chunks to Elasticsearch via helpers.bulk().

        raise_on_error=False so that a handful of malformed
        documents don't abort the entire load; every individual
        failure is captured and reported instead.
        """

        logger.info(
            "Starting bulk index into '%s' (batch size %d)...",
            ELASTIC_INDEX_NAME,
            BULK_CHUNK_SIZE,
        )

        success_count, errors = helpers.bulk(
            self.client,
            self.generate_bulk_actions(chunks),
            chunk_size=BULK_CHUNK_SIZE,
            raise_on_error=False,
            stats_only=False,
        )

        return success_count, errors

    # -----------------------------------------------------------------

    def refresh_index(self):
        """
        Force a refresh so newly-indexed documents are immediately
        visible to the verification count() call below. (ES
        normally refreshes automatically every ~1s, but an explicit
        refresh removes any race in a script that indexes then
        immediately verifies.)
        """

        self.client.indices.refresh(index=ELASTIC_INDEX_NAME)

    # -----------------------------------------------------------------

    def verify_document_count(self) -> int:

        self.refresh_index()

        count_response = self.client.count(index=ELASTIC_INDEX_NAME)

        indexed_count = count_response["count"]

        return indexed_count

    # -----------------------------------------------------------------

    def print_summary(self, indexed_count: int):

        logger.info("=" * 65)
        logger.info("Bulk Indexing Summary")
        logger.info("=" * 65)

        logger.info("Source File        : %s", CHUNKS_WITH_EMBEDDINGS_JSON)
        logger.info("Index Name         : %s", ELASTIC_INDEX_NAME)
        logger.info("Chunks In File      : %d", self.total_chunks)
        logger.info("Skipped (invalid)  : %d", self.skipped_chunks)
        logger.info("Indexed Successfully: %d", self.indexed_success)
        logger.info("Bulk Errors        : %d", len(self.indexed_errors))
        logger.info("-" * 65)
        logger.info(
            "Verified Document Count in Elasticsearch: %d",
            indexed_count,
        )

        if indexed_count == self.indexed_success:
            logger.info(
                "%d documents indexed successfully.",
                indexed_count,
            )
        else:
            logger.warning(
                "Mismatch: bulk() reported %d successes but "
                "Elasticsearch count() returned %d. Check the "
                "errors list above and consider re-running.",
                self.indexed_success,
                indexed_count,
            )

        if self.indexed_errors:

            logger.warning(
                "First %d bulk errors (of %d total):",
                min(5, len(self.indexed_errors)),
                len(self.indexed_errors),
            )

            for error in self.indexed_errors[:5]:
                logger.warning("  %s", error)

        logger.info("=" * 65)

    # -----------------------------------------------------------------

    def run(self):

        logger.info("=" * 65)
        logger.info("Indexing Documents into Elasticsearch")
        logger.info("=" * 65)

        chunks = self.load_chunks()

        success_count, errors = self.run_bulk_index(chunks)

        self.indexed_success = success_count

        self.indexed_errors = errors

        indexed_count = self.verify_document_count()

        self.print_summary(indexed_count)


# ---------------------------------------------------------------------


def main():

    indexer = DocumentIndexer()

    indexer.run()


if __name__ == "__main__":
    main()