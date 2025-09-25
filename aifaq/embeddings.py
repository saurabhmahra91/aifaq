"""
Embedding generation and management using LiteLLM.
"""

import logging
import math
import os

import litellm
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")


class EmbeddingManager:
    """
    Manages text embeddings using LiteLLM for OpenAI integration.
    """

    def __init__(self, model: str = EMBEDDING_MODEL):
        """
        Initialize the embedding manager.

        Args:
            model: The embedding model to use
        """
        self.model = model
        self._embedding_cache = {}
        logger.info("Initialized EmbeddingManager with model: %s", model)

    def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If embedding generation fails
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        try:
            response = litellm.embedding(model=self.model, input=[text])
            embedding = response.data[0]["embedding"]
            self._embedding_cache[text] = embedding
            logger.debug("Generated embedding for text: %s...", text[:50])
            return embedding

        except Exception as e:
            logger.error("Failed to generate embedding: %s", str(e))
            raise

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Check cache first
        cached_embeddings = []
        texts_to_process = []

        for text in texts:
            if text in self._embedding_cache:
                cached_embeddings.append(self._embedding_cache[text])
            else:
                texts_to_process.append(text)
                cached_embeddings.append(None)

        # Process uncached texts
        if texts_to_process:
            try:
                response = litellm.embedding(model=self.model, input=texts_to_process)

                new_embeddings = [data["embedding"] for data in response.data]

                # Update cache and result list
                new_idx = 0
                for i, text in enumerate(texts):
                    if cached_embeddings[i] is None:
                        embedding = new_embeddings[new_idx]
                        self._embedding_cache[text] = embedding
                        cached_embeddings[i] = embedding
                        new_idx += 1

                logger.info("Generated embeddings for %d texts", len(texts_to_process))

            except Exception as e:
                logger.error("Failed to generate batch embeddings: %s", str(e))
                raise

        return cached_embeddings

    def compute_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        # Compute dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2, strict=False))

        # Compute norms
        norm1 = math.sqrt(sum(a * a for a in embedding1))
        norm2 = math.sqrt(sum(b * b for b in embedding2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def clear_cache(self) -> None:
        """
        Clear the embedding cache.
        """
        self._embedding_cache.clear()
        logger.info("Cleared embedding cache")

    def get_cache_size(self) -> int:
        """
        Get the current cache size.

        Returns:
            Number of cached embeddings
        """
        return len(self._embedding_cache)
