"""
===========================================================
Legal Document Chunker

Purpose
-------
Convert page-wise cleaned text (clean_pages.json) into
document-level, word-based chunks suitable for embedding
and retrieval in a legal RAG system.

Why not page-wise chunking?
----------------------------
Legal sections frequently continue across page boundaries.
Chunking page-by-page breaks sentences and legal clauses in
the middle. Instead, this module:

    1. Groups pages by document_id
    2. Sorts pages by page_number
    3. Merges all page text into one continuous document,
       while remembering which page every word came from
    4. Splits the merged text into fixed-size word chunks
       with overlap
    5. Maps each chunk back to its page_start / page_end

Input
-----
clean_pages.json

Output
------
chunks.json

Author : Shyam Suman
===========================================================
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

from config import (
    CLEAN_PAGES_JSON,
    CHUNKS_JSON,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_TRAILING_CHUNK_SIZE,
)

# =====================================================
# Logger
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class LegalChunker:
    """
    Performs document-level, word-based chunking with page mapping.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap

        self.step = chunk_size - chunk_overlap

    # -------------------------------------------------
    # Step 1 : Merge pages into a single word stream,
    #          while keeping a parallel list of page
    #          numbers, one per word.
    # -------------------------------------------------

    def build_word_page_map(
        self,
        pages: List[Dict],
    ) -> Tuple[List[str], List[int], List[str]]:
        """
        Merge all pages of a document into one continuous
        sequence of words, tracking the originating page
        number of every word.

        Returns
        -------
        words       : list of words across the whole document
        word_pages  : list of page numbers, same length as words
        """

        words: List[str] = []

        word_pages: List[int] = []

        word_page_ids: List[str] = []

        for page in pages:

            page_number = page.get("page_number")
            page_id = page.get("page_id")

            text = page.get("text", "")

            if not isinstance(text, str) or not text.strip():
                continue

            page_words = text.split()

            words.extend(page_words)

            word_pages.extend([page_number] * len(page_words))

            word_page_ids.extend([page_id] * len(page_words))

        return words, word_pages, word_page_ids

    # -------------------------------------------------
    # Step 2 : Slide a window over the word stream to
    #          produce overlapping chunks.
    # -------------------------------------------------

    def split_into_chunks(
        self,
        words: List[str],
        word_pages: List[int],
        word_page_ids: List[str],
    ) -> List[Dict]:
        """
        Slide a fixed-size, overlapping window over the word
        stream and produce raw chunk dicts containing:

            text, word_count, character_count,
            page_start, page_end
        """

        raw_chunks: List[Dict] = []

        total_words = len(words)

        if total_words == 0:
            return raw_chunks

        start = 0

        while start < total_words:

            end = min(start + self.chunk_size, total_words)

            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            page_start = word_pages[start]

            page_end = word_pages[end - 1]

            page_ids = list(dict.fromkeys(word_page_ids[start:end]))

            raw_chunks.append(
                {
                    "text": chunk_text,
                    "word_count": len(chunk_words),
                    "character_count": len(chunk_text),
                    "page_start": page_start,
                    "page_end": page_end,
                    "page_ids": page_ids,
                }
            )

            # Stop once we've reached the end of the document
            if end == total_words:
                break

            start += self.step

        raw_chunks = self._merge_small_trailing_chunk(raw_chunks)

        return raw_chunks

    # -------------------------------------------------
    # Step 3 : Avoid a tiny, low-value final chunk by
    #          merging it into the previous one.
    # -------------------------------------------------

    def _merge_small_trailing_chunk(
        self,
        raw_chunks: List[Dict],
    ) -> List[Dict]:
        """
        If the last chunk is very small (e.g. only a handful
        of leftover words), merge it into the previous chunk
        instead of storing it as a near-useless standalone
        chunk.
        """

        if len(raw_chunks) < 2:
            return raw_chunks

        last_chunk = raw_chunks[-1]

        if last_chunk["word_count"] >= MIN_TRAILING_CHUNK_SIZE:
            return raw_chunks

        second_last_chunk = raw_chunks[-2]

        merged_text = second_last_chunk["text"] + " " + last_chunk["text"]

        merged_chunk = {
            "text": merged_text,
            "word_count": len(merged_text.split()),
            "character_count": len(merged_text),
            "page_start": second_last_chunk["page_start"],
            "page_end": last_chunk["page_end"],
            "page_ids": list(dict.fromkeys(second_last_chunk["page_ids"] +last_chunk["page_ids"])),
        }

        raw_chunks = raw_chunks[:-2] + [merged_chunk]

        return raw_chunks

    # -------------------------------------------------
    # Step 4 : Attach metadata + unique IDs to raw chunks
    # -------------------------------------------------

    def build_chunk_records(
        self,
        raw_chunks: List[Dict],
        document_meta: Dict,
    ) -> List[Dict]:
        """
        Convert raw chunk dicts into fully-formed chunk
        records, inheriting document-level metadata and
        adding chunk-level metadata.
        """

        chunk_records: List[Dict] = []

        for index, raw_chunk in enumerate(raw_chunks):

            chunk_record = {
                #"chunk_id": str(uuid.uuid4()),
                "chunk_id": (f"{document_meta['document_id']}"f"_CHUNK_{index+1:04d}"),
                "document_id": document_meta.get("document_id"),
                "document_name": document_meta.get("document_name"),
                "document_type": document_meta.get("document_type"),
                "source_file": document_meta.get("source_file"),
                "chunk_index": index + 1,
                "page_start": raw_chunk["page_start"],
                "page_end": raw_chunk["page_end"],
                "page_ids": raw_chunk["page_ids"],
                "character_count": raw_chunk["character_count"],
                "word_count": raw_chunk["word_count"],
                "text": raw_chunk["text"],
                "chunked_at": datetime.now().isoformat(),
            }

            chunk_records.append(chunk_record)

        return chunk_records

    # -------------------------------------------------
    # Public entry point : chunk a single document
    # -------------------------------------------------

    def chunk_document(
        self,
        pages: List[Dict],
    ) -> List[Dict]:
        """
        Full pipeline for a single document:
        merge -> split -> attach metadata.
        """

        if not pages:
            return []

        # Sort pages by page_number to guarantee correct order
        sorted_pages = sorted(
            pages,
            key=lambda page: page.get("page_number", 0),
        )

        document_meta = {
            "document_id": sorted_pages[0].get("document_id"),
            "document_name": sorted_pages[0].get("document_name"),
            "document_type": sorted_pages[0].get("document_type"),
            "source_file": sorted_pages[0].get("source_file"),
        }

        words, word_pages, word_page_ids = self.build_word_page_map(sorted_pages)

        raw_chunks = self.split_into_chunks(words, word_pages, word_page_ids)

        chunk_records = self.build_chunk_records(
            raw_chunks,
            document_meta,
        )

        return chunk_records


class ChunkerPipeline:
    """
    Orchestrates the full chunking pipeline:
    load -> group -> chunk -> save -> summarize.
    """

    def __init__(self):

        self.chunker = LegalChunker()

        self.total_pages = 0

        self.total_documents = 0

        self.all_chunks: List[Dict] = []

        self.chunks_per_document: Dict[str, int] = {}

    # -------------------------------------------------

    def load_pages(self) -> List[Dict]:

        logger.info("Loading clean_pages.json")

        with open(
            CLEAN_PAGES_JSON,
            "r",
            encoding="utf-8",
        ) as file:

            pages = json.load(file)

        self.total_pages = len(pages)

        logger.info(f"{self.total_pages} pages loaded.")

        return pages

    # -------------------------------------------------

    def group_pages_by_document(
        self,
        pages: List[Dict],
    ) -> Dict[str, List[Dict]]:
        """
        Group flat list of pages into { document_id: [pages] }.
        """

        logger.info("Grouping pages by document_id")

        grouped: Dict[str, List[Dict]] = {}

        for page in pages:

            document_id = page.get("document_id")

            if document_id is None:
                logger.warning(
                    f"Skipping page with missing document_id: "
                    f"{page.get('page_id')}"
                )
                continue

            grouped.setdefault(document_id, []).append(page)

        self.total_documents = len(grouped)

        logger.info(
            f"Found {self.total_documents} unique documents."
        )

        return grouped

    # -------------------------------------------------

    def chunk_all_documents(
        self,
        grouped_pages: Dict[str, List[Dict]],
    ):

        logger.info("Chunking documents...")

        for document_id, pages in grouped_pages.items():

            document_name = pages[0].get("document_name", document_id)

            try:
                chunks = self.chunker.chunk_document(pages)

            except Exception as error:

                logger.error(
                    f"Failed to chunk document '{document_name}' "
                    f"({document_id}): {error}"
                )
                continue

            self.all_chunks.extend(chunks)

            self.chunks_per_document[document_name] = len(chunks)

            logger.info(
                f"  {document_name}: {len(pages)} pages -> "
                f"{len(chunks)} chunks"
            )

    # -------------------------------------------------

    def save_chunks(self):

        logger.info("Saving chunks.json")

        with open(
            CHUNKS_JSON,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.all_chunks,
                file,
                indent=4,
                ensure_ascii=False,
            )

    # -------------------------------------------------

    def print_summary(self):

        total_chunks = len(self.all_chunks)

        avg_words = (
            sum(chunk["word_count"] for chunk in self.all_chunks)
            / total_chunks
            if total_chunks
            else 0
        )

        logger.info("=" * 60)

        logger.info("Chunking Summary")

        logger.info("=" * 60)

        logger.info(f"Documents Processed : {self.total_documents}")

        logger.info(f"Pages Processed     : {self.total_pages}")

        logger.info(f"Total Chunks        : {total_chunks}")

        logger.info(f"Avg Words / Chunk    : {avg_words:.1f}")

        logger.info(f"Chunk Size (words)  : {self.chunker.chunk_size}")

        logger.info(f"Chunk Overlap (words): {self.chunker.chunk_overlap}")

        logger.info(f"Output File         : {CHUNKS_JSON}")

        logger.info("-" * 60)

        logger.info("Chunks per document:")

        for document_name, count in self.chunks_per_document.items():
            logger.info(f"  {document_name}: {count} chunks")

        logger.info("=" * 60)

    # -------------------------------------------------

    def run(self):

        pages = self.load_pages()

        grouped_pages = self.group_pages_by_document(pages)

        self.chunk_all_documents(grouped_pages)

        self.save_chunks()

        self.print_summary()


def main():

    pipeline = ChunkerPipeline()

    pipeline.run()


if __name__ == "__main__":

    main()