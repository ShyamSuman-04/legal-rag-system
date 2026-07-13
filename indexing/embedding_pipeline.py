"""
embedding_pipeline.py

Reads chunks_with_metadata.json,
generates embeddings for every chunk,
and saves chunks_with_embeddings.json.

Author: Shyam Suman
Project: US Tax & Legal RAG System
"""

import json
import logging
from pathlib import Path

from config import (
    CHUNKS_WITH_METADATA_JSON,
    CHUNKS_WITH_EMBEDDINGS_JSON,
)

from indexing.embedding_generator import EmbeddingGenerator

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class EmbeddingPipeline:

    def __init__(self):

        self.generator = EmbeddingGenerator()

    # -------------------------------------------------------------

    def load_chunks(self):

        logger.info("Loading chunks_with_metadata.json")

        with open(
            CHUNKS_WITH_METADATA_JSON,
            "r",
            encoding="utf-8"
        ) as f:

            chunks = json.load(f)

        logger.info(
            "%d chunks loaded.",
            len(chunks)
        )

        return chunks

    # -------------------------------------------------------------

    def extract_texts(self, chunks):

        texts = []

        for chunk in chunks:

            text = chunk.get("text", "")

            texts.append(text)

        return texts

    # -------------------------------------------------------------

    def attach_embeddings(
        self,
        chunks,
        embeddings
    ):

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            chunk["embedding"] = embedding

        return chunks

    # -------------------------------------------------------------

    def save_chunks(self, chunks):

        output_path = Path(
            CHUNKS_WITH_EMBEDDINGS_JSON
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                chunks,
                f,
                indent=2,
                ensure_ascii=False
            )

        logger.info(
            "Saved embeddings to\n%s",
            output_path
        )

    # -------------------------------------------------------------

    def run(self):

        logger.info("=" * 60)
        logger.info("Embedding Pipeline Started")
        logger.info("=" * 60)

        chunks = self.load_chunks()

        texts = self.extract_texts(chunks)

        embeddings = self.generator.generate_embeddings(
            texts
        )

        chunks = self.attach_embeddings(
            chunks,
            embeddings
        )

        self.save_chunks(chunks)

        logger.info("=" * 60)
        logger.info("Embedding Pipeline Summary")
        logger.info("=" * 60)
        logger.info(
            "Chunks Processed : %d",
            len(chunks)
        )

        logger.info(
            "Output File : %s",
            CHUNKS_WITH_EMBEDDINGS_JSON
        )

        logger.info("=" * 60)


# ---------------------------------------------------------------------

def main():

    pipeline = EmbeddingPipeline()

    pipeline.run()


if __name__ == "__main__":
    main()