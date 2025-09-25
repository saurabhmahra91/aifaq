"""
Response generation using LiteLLM for OpenAI integration.
"""

import logging
import os

import litellm
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-3.5-turbo")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))


class ResponseGenerator:
    """
    Generates natural language responses using retrieved FAQ context.
    """

    def __init__(self, model: str = CHAT_MODEL, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE):
        """
        Initialize the response generator.

        Args:
            model: LLM model to use for generation
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        logger.info("Initialized ResponseGenerator with model: %s", model)

    def generate_response(self, query: str, retrieved_faqs: list[dict], response_mode: str = "helpful") -> str:
        """
        Generate a natural language response based on query and retrieved FAQs.

        Args:
            query: User's original query
            retrieved_faqs: List of retrieved FAQ dictionaries with similarity scores
            response_mode: Response style ("helpful", "concise", "detailed")

        Returns:
            Generated response string
        """
        try:
            # Build context from retrieved FAQs
            context = self._build_context(retrieved_faqs)

            # Create prompt based on response mode
            prompt = self._create_prompt(query, context, response_mode)

            # Generate response using LiteLLM
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt(response_mode)},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            generated_text = response.choices[0].message.content.strip()

            logger.info("Generated response for query: %s", query[:50])
            return generated_text

        except Exception as e:
            logger.error("Failed to generate response: %s", str(e))
            return self._get_fallback_response(query, retrieved_faqs)

    def _build_context(self, retrieved_faqs: list[dict]) -> str:
        """
        Build context string from retrieved FAQs.

        Args:
            retrieved_faqs: List of FAQ dictionaries with similarity scores

        Returns:
            Formatted context string
        """
        if not retrieved_faqs:
            return "No relevant FAQs found."

        context_parts = []
        for i, item in enumerate(retrieved_faqs, 1):
            faq = item["faq"]
            similarity = item["similarity"]

            context_parts.append(f"FAQ {i} (similarity: {similarity:.2f}):\nQ: {faq['question']}\nA: {faq['answer']}\n")

        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str, response_mode: str) -> str:
        """
        Create the prompt for the LLM.

        Args:
            query: User query
            context: Retrieved FAQ context
            response_mode: Response style

        Returns:
            Formatted prompt string
        """
        if response_mode == "concise":
            instruction = "Provide a brief, direct answer"
        elif response_mode == "detailed":
            instruction = "Provide a comprehensive, detailed response"
        else:  # helpful
            instruction = "Provide a helpful, friendly response"

        return f"""User Query: {query}

Relevant FAQ Information:
{context}

{instruction} based on the FAQ information above. If the FAQs don't fully answer the question, acknowledge what you can answer and suggest they may need additional help."""

    def _get_system_prompt(self, response_mode: str) -> str:
        """
        Get system prompt based on response mode.

        Args:
            response_mode: Response style

        Returns:
            System prompt string
        """
        base_prompt = """You are a helpful FAQ assistant. Your role is to answer user questions based on the provided FAQ information."""

        if response_mode == "concise":
            return base_prompt + " Keep your responses brief and to the point."
        elif response_mode == "detailed":
            return base_prompt + " Provide thorough, comprehensive answers with relevant details."
        else:  # helpful
            return base_prompt + " Be friendly, helpful, and conversational in your responses."

    def _get_fallback_response(self, query: str, retrieved_faqs: list[dict]) -> str:
        """
        Generate fallback response when LLM fails.

        Args:
            query: User query
            retrieved_faqs: Retrieved FAQs

        Returns:
            Fallback response string
        """
        if not retrieved_faqs:
            return (
                "I couldn't find any relevant information in our FAQ database "
                "for your question. Please contact our support team for assistance."
            )

        # Use the best matching FAQ directly
        best_faq = retrieved_faqs[0]["faq"]
        similarity = retrieved_faqs[0]["similarity"]

        return (
            f"Based on our FAQ database (similarity: {similarity:.2f}), "
            f"here's the most relevant information:\n\n"
            f"Q: {best_faq['question']}\n"
            f"A: {best_faq['answer']}"
        )

    def generate_response_with_confidence(
        self, query: str, retrieved_faqs: list[dict], response_mode: str = "helpful"
    ) -> dict:
        """
        Generate response with confidence metrics.

        Args:
            query: User's original query
            retrieved_faqs: List of retrieved FAQ dictionaries
            response_mode: Response style

        Returns:
            Dictionary with response and confidence metrics
        """
        response = self.generate_response(query, retrieved_faqs, response_mode)

        # Calculate confidence based on similarity scores
        if retrieved_faqs:
            max_similarity = max(item["similarity"] for item in retrieved_faqs)
            avg_similarity = sum(item["similarity"] for item in retrieved_faqs) / len(retrieved_faqs)
            confidence = (max_similarity + avg_similarity) / 2
        else:
            confidence = 0.0

        return {
            "response": response,
            "confidence": confidence,
            "num_sources": len(retrieved_faqs),
            "max_similarity": max_similarity if retrieved_faqs else 0.0,
        }
