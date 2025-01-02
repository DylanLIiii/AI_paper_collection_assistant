from typing import TypeVar, Generator, List
from requests import Session
import time
from tqdm import tqdm

T = TypeVar("T")

def batched(items: List[T], batch_size: int) -> List[List[T]]:
    """Takes a list and returns a list of lists with batch_size."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

def argsort(seq):
    """Native Python version of an 'argsort'."""
    return sorted(range(len(seq)), key=seq.__getitem__)

def get_batch(
    session: Session,
    ids: List[str],
    S2_API_KEY: str,
    endpoint: str,
    fields: str = "paperId,title",
    **kwargs,
) -> List[dict]:
    """Generic function to get a batch of papers or authors."""
    params = {
        "fields": fields,
        **kwargs,
    }
    headers = {"X-API-KEY": S2_API_KEY} if S2_API_KEY else {}
    body = {"ids": ids}

    with session.post(
        f"https://api.semanticscholar.org/graph/v1/{endpoint}/batch",
        params=params,
        headers=headers,
        json=body,
    ) as response:
        response.raise_for_status()
        return response.json()

@retry(tries=3, delay=2.0)
def get_one_author(session, author: str, S2_API_KEY: str) -> str:
    """Query the Semantic Scholar API for a single author."""
    params = {"query": author, "fields": "authorId,name,hIndex", "limit": "10"}
    headers = {"X-API-KEY": S2_API_KEY} if S2_API_KEY else {}
    with session.get(
        "https://api.semanticscholar.org/graph/v1/author/search",
        params=params,
        headers=headers,
    ) as response:
        try:
            response.raise_for_status()
            response_json = response.json()
            return response_json["data"] if len(response_json["data"]) >= 1 else None
        except Exception as ex:
            print(f"Exception happened: {ex}")
            return None