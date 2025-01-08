from datetime import datetime
from typing import List, Optional, Dict, Any
from loguru import logger
from paper_assistant.core.base_paper import BasePaper


class EmbeddingPaper(BasePaper):
    """Paper class specialized for embedding operations."""
    
    # Additional required attributes specific to embedding papers
    REQUIRED_ATTRIBUTES = {
        **BasePaper.REQUIRED_ATTRIBUTES,
        'categories': None,  # Paper categories/topics
        'versions': [],     # List of paper versions
        'update_date': None # Last update date
    }

    def __init__(self, **kwargs):
        """
        Initialize an embedding paper.
        Includes additional functionality for handling versions and categories.
        """
        super().__init__(**kwargs)
        
        # Handle update date if versions exist
        if hasattr(self, "versions") and self.versions:
            self.update_date = self._get_latest_version_date()

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'EmbeddingPaper':
        """Create an EmbeddingPaper from a JSON dictionary."""
        return cls(**json_data)

    def is_category(self, category: str) -> bool:
        """Check if the paper belongs to a specific category."""
        return getattr(self, "categories", None) == category

    def was_updated_after(self, compare_date: datetime) -> bool:
        """
        Check if the paper was updated after a given date.

        Args:
            compare_date: datetime object to compare against

        Returns:
            bool: True if paper was updated after the given date
        """
        if not hasattr(self, "update_date"):
            return False
        return self.update_date >= compare_date

    def get_text_for_embedding(self) -> str:
        """
        Get the text representation of the paper for embedding.
        Override this method to customize what text is used for embeddings.
        """
        fields = ["title", "abstract"]
        text_parts = [str(getattr(self, field, "")) for field in fields]
        return " ".join(text_parts)
