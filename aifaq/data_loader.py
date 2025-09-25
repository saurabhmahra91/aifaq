"""
FAQ data loading and preprocessing module.
"""

import csv
import logging

logger = logging.getLogger(__name__)

DEFAULT_FAQ_PATH = "faqs.csv"


def load_faq_data(file_path: str = DEFAULT_FAQ_PATH) -> list[dict[str, str]]:
    """
    Load FAQ data from CSV file and return as list of dictionaries.

    Args:
        file_path: Path to the CSV file containing FAQ data

    Returns:
        List of dictionaries with 'question' and 'answer' keys

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file has invalid format
    """
    try:
        with open(file_path, encoding='utf-8') as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                raise ValueError("CSV file is empty or has no headers")

            if "Question" not in reader.fieldnames or "Answer" not in reader.fieldnames:
                raise ValueError("CSV file must contain 'Question' and 'Answer' columns")

            logger.info("Loaded FAQ data from %s", file_path)

            # Load and clean data
            faq_data = []
            for row in reader:
                # Skip rows with missing data
                if not row.get("Question") or not row.get("Answer"):
                    continue

                # Clean and normalize the data
                question = row["Question"].strip()
                answer = row["Answer"].strip()

                # Skip empty entries after stripping
                if question and answer:
                    faq_data.append({"question": question, "answer": answer})

        logger.info("Processed %d FAQ entries", len(faq_data))
        return faq_data

    except FileNotFoundError:
        logger.error("FAQ file not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Error loading FAQ data: %s", str(e))
        raise ValueError(f"Failed to load FAQ data: {e}") from e


def _validate_faq_entry(entry: dict[str, str]) -> bool:
    """
    Validate that an FAQ entry has required fields and non-empty content.

    Args:
        entry: Dictionary with 'question' and 'answer' keys

    Returns:
        True if entry is valid, False otherwise
    """
    return (
        isinstance(entry, dict)
        and "question" in entry
        and "answer" in entry
        and len(entry["question"].strip()) > 0
        and len(entry["answer"].strip()) > 0
    )


def get_faq_questions(faq_data: list[dict[str, str]]) -> list[str]:
    """
    Extract just the questions from FAQ data.

    Args:
        faq_data: List of FAQ dictionaries

    Returns:
        List of question strings
    """
    return [faq["question"] for faq in faq_data if _validate_faq_entry(faq)]


def get_faq_count(faq_data: list[dict[str, str]]) -> int:
    """
    Get the count of valid FAQ entries.

    Args:
        faq_data: List of FAQ dictionaries

    Returns:
        Number of valid FAQ entries
    """
    return len([faq for faq in faq_data if _validate_faq_entry(faq)])
