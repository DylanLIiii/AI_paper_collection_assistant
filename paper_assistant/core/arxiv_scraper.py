import configparser
import json
from datetime import datetime, timedelta
from html import unescape
from typing import List, Optional
import re
import arxiv
import feedparser
from loguru import logger
from paper_assistant.core.base_paper import BasePaper


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class ArxivPaper(BasePaper):
    """Specialized paper class for ArXiv papers with ArXiv-specific functionality."""

    # Additional required attributes specific to ArXiv papers
    REQUIRED_ATTRIBUTES = {
        **BasePaper.REQUIRED_ATTRIBUTES,
        "arxiv_id": None,
    }

    def __init__(
        self,
        arxiv_id=None,
        ARXIVID=None,
        title=None,
        abstract=None,
        authors=None,
        url=None,
        comment=None,
        COMMENT=None,
        relevance=None,
        RELEVANCE=None,
        novelty=None,
        NOVELTY=None,
        **kwargs,
    ):
        """
        Initialize an ArXiv paper.
        Handles both camelCase and uppercase variants of parameters for backwards compatibility.
        """
        # Normalize parameters
        normalized_kwargs = {
            "arxiv_id": arxiv_id or ARXIVID,
            "title": title,
            "abstract": abstract,
            "authors": authors or [],
            "url": url
            or (arxiv_id and f"https://arxiv.org/abs/{arxiv_id}")
            or (ARXIVID and f"https://arxiv.org/abs/{ARXIVID}"),
            "comment": comment or COMMENT,
            "relevance": relevance or RELEVANCE,
            "novelty": novelty or NOVELTY,
            **kwargs,
        }

        # Remove None values
        normalized_kwargs = {
            k: v for k, v in normalized_kwargs.items() if v is not None
        }

        super().__init__(**normalized_kwargs)

    @classmethod
    def from_arxiv_result(cls, result) -> "ArxivPaper":
        """Create an ArxivPaper from an arxiv API result."""
        return cls(
            arxiv_id=result.get_short_id()[:10],
            title=result.title,
            abstract=unescape(re.sub("\n", " ", result.summary)),
            authors=[author.name for author in result.authors],
            url=f"https://arxiv.org/abs/{result.get_short_id()[:10]}",
        )

    @classmethod
    def from_feed_entry(cls, entry) -> "ArxivPaper":
        """Create an ArxivPaper from a feedparser entry."""
        # Clean up the data
        authors = [
            unescape(re.sub("<[^<]+?>", "", author)).strip()
            for author in entry.author.replace("\n", ", ").split(",")
        ]
        summary = unescape(re.sub("\n", " ", re.sub("<[^<]+?>", "", entry.summary)))
        title = re.sub("\(arXiv:[0-9]+\.[0-9]+v[0-9]+ \[.*\]\)$", "", entry.title)
        arxiv_id = entry.link.split("/")[-1]

        return cls(
            arxiv_id=arxiv_id,
            title=title,
            abstract=summary,
            authors=authors,
            url=entry.link,
        )


def is_earlier(ts1, ts2):
    """Compare two arxiv IDs, returns true if ts1 is older than ts2."""
    return int(ts1.replace(".", "")) < int(ts2.replace(".", ""))


def get_papers_from_arxiv_api(
    area: Optional[str] = None, 
    timestamp = None, 
    last_id = None, 
    days: int = 1,
    arxiv_ids: Optional[List[str]] = None
) -> List[ArxivPaper]:
    """
    Get papers from ArXiv API that are newer than the last_id or match specific arxiv IDs.
    
    Args:
        area: Subject area to search for papers
        timestamp: Timestamp to start search from
        last_id: Only return papers newer than this ID
        days: Number of days to look back from timestamp
        arxiv_ids: List of specific arxiv IDs to fetch
    """
    if arxiv_ids:
        logger.info(f"Getting papers from ArXiv API for IDs: {arxiv_ids}")
        search = arxiv.Search(
            id_list=arxiv_ids,
            max_results=None
        )
    else:
        logger.info(
            f"Getting papers from ArXiv API for {area}. You can modify the timestamp and days to get papers from the last week or month from the timestamp."
        )
        end_date = timestamp
        start_date = timestamp - timedelta(days=days)
        search = arxiv.Search(
            query=f"({area}) AND submittedDate:[{start_date.strftime('%Y%m%d')}* TO {end_date.strftime('%Y%m%d')}*]",
            max_results=None,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

    results = list(arxiv.Client().results(search))
    api_papers = []

    for result in results:
        new_id = result.get_short_id()[:10]
        if arxiv_ids or (last_id and is_earlier(last_id, new_id)):
            paper = ArxivPaper.from_arxiv_result(result)
            api_papers.append(paper)

    return api_papers


def get_papers_from_arxiv_rss(area: str, config: Optional[dict]) -> List[ArxivPaper]:
    """Get papers from ArXiv RSS feed."""

    logger.info(
        f"Getting papers from ArXiv RSS feed for {area}. This method only get papers from the last day. If you want to get papers from the last week, you should use the API method."
    )
    # Get feed with timestamp to avoid duplicates
    updated = datetime.utcnow() - timedelta(days=1)
    updated_string = updated.strftime("%a, %d %b %Y %H:%M:%S GMT")
    feed = feedparser.parse(
        f"http://export.arxiv.org/rss/{area}", modified=updated_string
    )

    # Handle no new papers case
    if feed.status == 304:
        if (config is not None) and config["OUTPUT"]["debug_messages"]:
            print(f"No new papers since {updated_string} for {area}")
        return [], None, None

    if len(feed.entries) == 0:
        print(f"No entries found for {area}")
        return [], None, None

    # Get timestamp and last ID
    last_id = feed.entries[0].link.split("/")[-1]
    timestamp = datetime.strptime(feed.feed["updated"], "%a, %d %b %Y %H:%M:%S +0000")

    # Process entries
    paper_list = []
    for entry in feed.entries:
        # Skip updated papers
        if entry["arxiv_announce_type"] != "new":
            continue

        # Check primary area if configured
        paper_area = entry.tags[0]["term"]
        if (area != paper_area) and (config["FILTERING"].getboolean("force_primary")):
            print(f"ignoring {entry.title}")
            continue

        paper = ArxivPaper.from_feed_entry(entry)
        paper_list.append(paper)

    return paper_list, timestamp, last_id


def merge_paper_list(paper_list, api_paper_list):
    """Merge two lists of papers, avoiding duplicates based on arxiv_id."""
    api_set = set(paper.arxiv_id for paper in api_paper_list)
    merged_paper_list = api_paper_list + [
        p for p in paper_list if p.arxiv_id not in api_set
    ]
    return merged_paper_list


def get_papers_from_arxiv_rss_api(category: str, config) -> List[ArxivPaper]:
    """Get papers from ArXiv RSS API for a given category."""
    max_results = int(config["FILTERING"]["max_results"])
    url = f"http://export.arxiv.org/rss/{category}?version=1.0"

    feed = feedparser.parse(url)
    paper_list = []

    for entry in feed.entries[:max_results]:
        paper = ArxivPaper.from_feed_entry(entry)
        paper_list.append(paper)

    return paper_list


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("configs/config.ini")
    paper_list, timestamp, last_id = get_papers_from_arxiv_rss("cs.CL", config)
    print(timestamp)
    api_paper_list = get_papers_from_arxiv_api("cs.CL", timestamp, last_id)
    merged_paper_list = merge_paper_list(paper_list, api_paper_list)
    print([paper.arxiv_id for paper in merged_paper_list])
    print([paper.arxiv_id for paper in paper_list])
    print([paper.arxiv_id for paper in api_paper_list])
    print("success")
