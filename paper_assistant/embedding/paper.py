from datetime import datetime
from loguru import logger
from typing import List, Optional, Dict, Any


class Paper:
    """Base class for all paper types with common functionality."""
    
    def __init__(self, **kwargs):
        """
        Initialize a paper with dynamic attributes.
        Common attributes include: id/arxiv_id, title, abstract, authors, url
        """
        # Dynamically assign all attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Ensure basic attributes exist
        self.id = getattr(self, 'id', None) or getattr(self, 'arxiv_id', None)
        self.title = getattr(self, 'title', None)
        self.abstract = getattr(self, 'abstract', None)
        self.authors = getattr(self, 'authors', [])
        self.url = getattr(self, 'url', None)
        
        # Update date handling
        if hasattr(self, "versions") and self.versions:
            self.update_date = self._get_latest_version_date()

    def _get_latest_version_date(self) -> Optional[datetime]:
        """
        Get the date of the latest version of the paper.

        Returns:
            datetime: The date of the latest version
        """
        try:
            versions = self.versions if hasattr(self, "versions") else []
            if not versions:
                logger.warning(f"No versions found for paper {self.id}")
                return self.update_date

            latest = versions[-1]
            if isinstance(latest, dict) and "created" in latest:
                date_str = latest["created"]
                try:
                    # First try the GMT format
                    return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
                except ValueError:
                    # If that fails, try the year-month-day format
                    return datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse date for paper {self.id}. Error: {e}")
            return None

    def __repr__(self) -> str:
        """Return a readable string representation of the object."""
        attrs = {
            key: getattr(self, key, None)
            for key in ["id", "title", "authors", "update_date"]
            if hasattr(self, key)
        }
        return f"Paper({', '.join(f'{k}={v}' for k, v in attrs.items())})"

    def __hash__(self) -> int:
        """Hash based on paper ID."""
        return hash(self.id)

    def __eq__(self, other) -> bool:
        """Equality based on paper ID."""
        if not isinstance(other, Paper):
            return False
        return self.id == other.id

    def is_category(self, category: str) -> bool:
        """Check if the paper belongs to a specific category."""
        return getattr(self, "categories", None) == category

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Paper object to a dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat() if value else None
            else:
                result[key] = value
        return result

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
