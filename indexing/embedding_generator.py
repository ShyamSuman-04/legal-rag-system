"""
embedding_generator.py

Generates dense vector embeddings for legal document chunks
using the SentenceTransformer embedding model.

Author: Shyam Suman
Project: US Tax & Legal RAG System
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    BGE_QUERY_PREFIX,
)

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Wrapper class around SentenceTransformer.

    Responsibilities:
    -----------------
    - Load embedding model
    - Generate document embeddings
    - Generate query embeddings
    - Generate embeddings in batches
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name

        logger.info("=" * 60)
        logger.info("Loading embedding model...")
        logger.info("Model : %s", model_name)

        try:
            self.model = SentenceTransformer(model_name)

        except Exception as e:
            logger.exception("Unable to load embedding model.")
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'."
            ) from e

        logger.info("Embedding model loaded successfully.")
        logger.info("=" * 60)

    # -----------------------------------------------------------------
    # Document Embedding
    # -----------------------------------------------------------------

    def embed_document(self, text: str) -> List[float]:
        """
        Generate embedding for a document chunk.

        Parameters
        ----------
        text : str

        Returns
        -------
        List[float]
        """

        if not text or not text.strip():
            return []

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    # -----------------------------------------------------------------
    # Query Embedding
    # -----------------------------------------------------------------

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        BGE models recommend using a query prefix.
        """

        if not query or not query.strip():
            return []

        query = BGE_QUERY_PREFIX + query

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    # -----------------------------------------------------------------
    # Batch Embedding
    # -----------------------------------------------------------------

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple document chunks.
        """

        if not texts:
            return []

        logger.info("Generating embeddings...")
        logger.info("Documents : %d", len(texts))
        logger.info("Batch Size : %d", batch_size)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        dimension = len(embeddings[0])

        if dimension != EMBEDDING_DIM:
            logger.warning(
                "Expected dimension %d but received %d",
                EMBEDDING_DIM,
                dimension,
            )

        logger.info("Embedding Dimension : %d", dimension)

        return embeddings.tolist()

    # -----------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------

    @staticmethod
    def cosine_similarity(
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two normalized vectors.

        Since embeddings are normalized,
        cosine similarity reduces to dot product.
        """

        if not embedding1 or not embedding2:
            return 0.0

        return sum(a * b for a, b in zip(embedding1, embedding2))


# ---------------------------------------------------------------------
# Local Test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    generator = EmbeddingGenerator()

    sample_document = (
        "Legal expenses incurred in carrying on a trade "
        "or business may qualify as deductible expenses."
    )

    sample_query = (
        "Can business legal expenses be deducted?"
    )

    print("\nGenerating document embedding...\n")

    document_embedding = generator.embed_document(sample_document)

    print("Embedding Length :", len(document_embedding))
    print("First 10 Values :")
    print(document_embedding[:10])

    print("\nGenerating query embedding...\n")

    query_embedding = generator.embed_query(sample_query)

    print("Embedding Length :", len(query_embedding))

    similarity = generator.cosine_similarity(
        document_embedding,
        query_embedding,
    )

    print("\nCosine Similarity :", round(similarity, 4))