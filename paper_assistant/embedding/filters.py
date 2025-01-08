from .paper import Paper
from datetime import datetime
from typing import Callable, List, Optional


class PaperFilter:
    """A composable filter for Paper objects"""

    def __init__(self, *conditions: Callable[[Paper], bool], operator: str = "AND"):
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

    def __call__(self, paper: Paper) -> bool:
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


def date_filter(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> PaperFilter:
    """Filter papers by date range"""
    conditions = []

    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        conditions.append(lambda p: p.was_updated_after(start))

    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d")
        conditions.append(lambda p: not p.was_updated_after(end))

    return PaperFilter(*conditions)


def version_filter(
    min_version: Optional[int] = None, max_version: Optional[int] = None
) -> PaperFilter:
    """Filter papers by version range"""
    conditions = []

    if min_version is not None:
        conditions.append(lambda p: p.version >= min_version)

    if max_version is not None:
        conditions.append(lambda p: p.version <= max_version)

    return PaperFilter(*conditions)


def title_contains(text: str, case_sensitive: bool = False) -> PaperFilter:
    """Filter papers by title containing text"""
    if not case_sensitive:
        text = text.lower()
        return PaperFilter(lambda p: text in p.title.lower())
    return PaperFilter(lambda p: text in p.title)


def has_doi() -> PaperFilter:
    """Filter papers that have a DOI"""
    return PaperFilter(lambda p: hasattr(p, "doi") and p.doi)


# Common filter combinations
def recent_papers(days: int = 30) -> PaperFilter:
    """Filter papers updated in the last N days"""
    from datetime import timedelta

    cutoff = datetime.now() - timedelta(days=days)
    return date_filter(cutoff.strftime("%Y-%m-%d"))


def popular_categories() -> PaperFilter:
    """Filter papers in popular categories"""
    popular = ["cs.CV", "cs.LG", "cs.AI", "cs.CL", "cs.NE"]
    return category_filter(popular)
