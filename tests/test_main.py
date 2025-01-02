import pytest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI_paper_collection_assistant.main import Paper, filter_papers_by_author, filter_papers_by_topic, sort_papers_by_score, format_output

@pytest.fixture
def sample_papers():
    return [
        Paper(authors=["Author A", "Author B"], title="Paper 1", abstract="Abstract 1", arxiv_id="1234.5678"),
        Paper(authors=["Author C"], title="Paper 2", abstract="Abstract 2", arxiv_id="2345.6789"),
        Paper(authors=["Author A"], title="Paper 3", abstract="Abstract 3", arxiv_id="3456.7890"),
    ]

def test_filter_papers_by_author(sample_papers):
    filtered = filter_papers_by_author(sample_papers, ["Author A"])
    assert len(filtered) == 2
    assert all("Author A" in paper.authors for paper in filtered)

def test_filter_papers_by_topic(sample_papers):
    with patch("main.get_paper_topics", return_value=["Topic 1", "Topic 2"]):
        filtered = filter_papers_by_topic(sample_papers, ["Topic 1"])
        assert len(filtered) == 3  # Mocked to return all papers

def test_sort_papers_by_score(sample_papers):
    with patch("main.calculate_relevance_score", side_effect=[0.8, 0.5, 0.9]):
        sorted_papers = sort_papers_by_score(sample_papers)
        assert sorted_papers[0].arxiv_id == "3456.7890"
        assert sorted_papers[-1].arxiv_id == "2345.6789"

def test_output_formatting(sample_papers):
    with patch("main.format_output", return_value="Formatted Output"):
        output = format_output(sample_papers)
        assert output == "Formatted Output"