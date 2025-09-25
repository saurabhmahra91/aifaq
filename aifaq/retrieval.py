"""
FAQ retrieval system with embedding-based similarity search.
"""

import logging

from .data_loader import load_faq_data
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class FAQRetriever:
    """
    Retrieves relevant FAQs using embedding-based similarity search.
    """

    def __init__(self, faq_data: list[dict[str, str]] | None = None):
        """
        Initialize the FAQ retriever.

        Args:
            faq_data: List of FAQ dictionaries, loads from CSV if None
        """
        self.embedding_manager = EmbeddingManager()

        if faq_data is None:
            faq_data = load_faq_data()

        self.faq_data = faq_data
        self.faq_embeddings = []

        # Precompute embeddings for all FAQ questions
        self._precompute_embeddings()

        logger.info("Initialized FAQRetriever with %d FAQs", len(self.faq_data))

    def _precompute_embeddings(self) -> None:
        """
        Precompute embeddings for all FAQ questions.
        """
        try:
            questions = [faq["question"] for faq in self.faq_data]
            self.faq_embeddings = self.embedding_manager.get_embeddings_batch(questions)
            logger.info("Precomputed embeddings for %d FAQ questions", len(questions))

        except Exception as e:
            logger.error("Failed to precompute embeddings: %s", str(e))
            raise

    def retrieve_similar_faqs(self, query: str, threshold: float = 0.7, top_k: int = 3) -> list[dict]:
        """
        Retrieve FAQs similar to the query.

        Args:
            query: User query
            threshold: Minimum similarity threshold
            top_k: Maximum number of FAQs to return

        Returns:
            List of dictionaries with FAQ data and similarity scores
        """
        try:
            # Get query embedding
            query_embedding = self.embedding_manager.get_embedding(query)

            # Compute similarities
            similarities = []
            for i, faq_embedding in enumerate(self.faq_embeddings):
                similarity = self.embedding_manager.compute_similarity(query_embedding, faq_embedding)

                if similarity >= threshold:
                    similarities.append({"faq": self.faq_data[i], "similarity": similarity, "index": i})

            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            result = similarities[:top_k]

            logger.info("Retrieved %d FAQs for query: %s", len(result), query[:50])

            return result

        except Exception as e:
            logger.error("Failed to retrieve similar FAQs: %s", str(e))
            return []

    def get_best_match(self, query: str, threshold: float = 0.7) -> dict | None:
        """
        Get the single best matching FAQ.

        Args:
            query: User query
            threshold: Minimum similarity threshold

        Returns:
            Best matching FAQ with similarity score or None
        """
        results = self.retrieve_similar_faqs(query, threshold, top_k=1)
        return results[0] if results else None

    def evaluate_threshold_performance(self, test_queries: list[str], threshold: float) -> dict:
        """
        Evaluate retrieval performance for a given threshold.

        Args:
            test_queries: List of test queries
            threshold: Threshold to evaluate

        Returns:
            Performance metrics dictionary
        """
        total_queries = len(test_queries)
        successful_retrievals = 0
        total_similarity = 0.0

        for query in test_queries:
            results = self.retrieve_similar_faqs(query, threshold, top_k=1)
            if results:
                successful_retrievals += 1
                total_similarity += results[0]["similarity"]

        success_rate = successful_retrievals / total_queries if total_queries > 0 else 0.0
        avg_similarity = total_similarity / successful_retrievals if successful_retrievals > 0 else 0.0

        return {
            "threshold": threshold,
            "success_rate": success_rate,
            "average_similarity": avg_similarity,
            "successful_retrievals": successful_retrievals,
            "total_queries": total_queries,
        }

    def get_faq_statistics(self) -> dict:
        """
        Get statistics about the FAQ database.

        Returns:
            Dictionary with FAQ statistics
        """
        return {
            "total_faqs": len(self.faq_data),
            "cache_size": self.embedding_manager.get_cache_size(),
            "avg_question_length": sum(len(faq["question"]) for faq in self.faq_data) / len(self.faq_data),
            "avg_answer_length": sum(len(faq["answer"]) for faq in self.faq_data) / len(self.faq_data),
        }

    def refresh_embeddings(self) -> None:
        """
        Refresh embeddings (useful if FAQ data changes).
        """
        self.embedding_manager.clear_cache()
        self._precompute_embeddings()
        logger.info("Refreshed all embeddings")
