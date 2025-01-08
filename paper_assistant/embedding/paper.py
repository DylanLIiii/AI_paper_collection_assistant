from datetime import datetime
from loguru import logger


class Paper:
    def __init__(self, **kwargs):
        """
        Dynamically assign attributes from JSON data.
        Update update_date based on the latest version's created date.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)  # 动态设置属性

        # Update update_date if versions exist
        if hasattr(self, "versions") and self.versions:
            self.update_date = self._get_latest_version_date()

    def _get_latest_version_date(self) -> datetime:
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
                    # First try the GMT format, sometimes that update version do not use GMT format, I do not know why
                    return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
                except ValueError:
                    # If that fails, try the year-month-day format
                    return datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse date for paper {self.id}. Error: {e}")
            return None

    def __repr__(self):
        """
        Return a readable string representation of the object as key-value pairs.
        """
        attrs = {
            key: getattr(self, key, None)
            for key in ["id", "title", "categories", "update_date"]
        }
        return f"Paper({', '.join(f'{k}={v}' for k, v in attrs.items())})"

    def is_category(self, category):
        """
        Check if the paper belongs to a specific category.
        """
        return getattr(self, "categories", None) == category

    def to_dict(self):
        """
        Convert the Paper object to a dictionary.
        Handles datetime serialization by converting to ISO format string.
        """
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
            date_str: Date string in 'YYYY-MM-DD' format or datetime object

        Returns:
            bool: True if paper was updated after the given date
        """
        if not hasattr(self, "update_date"):
            return False

        paper_date = self.update_date

        return paper_date >= compare_date
