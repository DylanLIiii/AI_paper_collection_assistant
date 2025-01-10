from .paper import EmbeddingPaper
from datetime import datetime
from typing import Callable, List


class PaperFilter:
    """A composable filter for Paper objects"""

    def __init__(
        self, *conditions: Callable[[EmbeddingPaper], bool], operator: str = "AND"
    ):
        """
        Initialize with one or more filter conditions

        Args:
            *conditions: One or more filter functions
            operator: 'AND' or 'OR' to combine conditions
        """
        self.conditions = conditions
        self.operator = operator.upper()

        if self.operator not in ["AND", "OR"]:
            raise ValueError("Operator must be 'AND' or 'OR'")

    def __call__(self, paper: EmbeddingPaper) -> bool:
        """Apply the filter to a paper"""
        if self.operator == "AND":
            return all(cond(paper) for cond in self.conditions)
        return any(cond(paper) for cond in self.conditions)

    def __and__(self, other):
        """Combine filters with AND"""
        return PaperFilter(*self.conditions, *other.conditions, operator="AND")

    def __or__(self, other):
        """Combine filters with OR"""
        return PaperFilter(*self.conditions, *other.conditions, operator="OR")


# Factory functions for common filters
def category_filter(categories: List[str]) -> PaperFilter:
    """Filter papers by category (accepts multiple categories)"""
    return PaperFilter(lambda p: any(p.is_category(cat) for cat in categories))


def date_filter(start_date: datetime) -> PaperFilter:
    """Filter papers by update date (after start_date)"""
    return PaperFilter(lambda p: p.was_updated_after(start_date))


def keyword_filter(
    keywords: List[str], fields: List[str] = ["title", "abstract"]
) -> PaperFilter:
    """
    Filter papers by keywords in specified fields.
    Case-insensitive partial matching.
    """

    def check_keywords(paper: EmbeddingPaper) -> bool:
        text = " ".join(str(getattr(paper, field, "")).lower() for field in fields)
        return any(kw.lower() in text for kw in keywords)

    return PaperFilter(check_keywords)
