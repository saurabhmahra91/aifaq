"""
AI FAQ Assistant - Core modules for FAQ retrieval and response generation.
"""

from .data_loader import get_faq_count, get_faq_questions, load_faq_data
from .embeddings import EmbeddingManager
from .generator import ResponseGenerator
from .retrieval import FAQRetriever
from .rl_optimizer import ThresholdOptimizer

__all__ = [
    "load_faq_data",
    "get_faq_questions",
    "get_faq_count",
    "EmbeddingManager",
    "FAQRetriever",
    "ResponseGenerator",
    "ThresholdOptimizer",
]
