import pytest
from datetime import datetime
from paper_assistant.core.arxiv_scraper import (
    ArxivPaper,
    is_earlier,
    get_papers_from_arxiv_api,
    get_papers_from_arxiv_rss,
    merge_paper_list
)

@pytest.fixture
def mock_arxiv_result():
    class MockResult:
        def get_short_id(self):
            return "2101.12345v1"
            
        @property
        def title(self):
            return "Test Paper"
            
        @property
        def summary(self):
            return "Test abstract"
            
        @property
        def authors(self):
            class MockAuthor:
                def __init__(self, name):
                    self.name = name
            return [MockAuthor("Author One"), MockAuthor("Author Two")]
    
    return MockResult()

@pytest.fixture
def mock_feed_entry():
    class MockEntry:
        def __init__(self):
            self.link = "https://arxiv.org/abs/2101.12345"
            self.title = "Test Paper (arXiv:2101.12345v1 [cs.CL])"
            self.summary = "Test abstract"
            self.author = "Author One, Author Two"
            self.tags = [{"term": "cs.CL"}]
            self.arxiv_announce_type = "new"
    
    return MockEntry()

class TestArxivPaper:
    def test_initialization(self):
        paper = ArxivPaper(
            arxiv_id="2101.12345",
            title="Test Paper",
            abstract="This is a test abstract",
            authors=["Author One", "Author Two"],
            url="https://arxiv.org/abs/2101.12345",
            comment="Test comment",
            relevance=0.8,
            novelty=0.9
        )
        
        assert paper.arxiv_id == "2101.12345"
        assert paper.title == "Test Paper"
        assert paper.abstract == "This is a test abstract"
        assert paper.authors == ["Author One", "Author Two"]
        assert paper.url == "https://arxiv.org/abs/2101.12345"
        assert paper.comment == "Test comment"
        assert paper.relevance == 0.8
        assert paper.novelty == 0.9

    def test_from_arxiv_result(self, mock_arxiv_result):
        paper = ArxivPaper.from_arxiv_result(mock_arxiv_result)
        
        assert paper.arxiv_id == "2101.12345"
        assert paper.title == "Test Paper"
        assert paper.abstract == "Test abstract"
        assert paper.authors == ["Author One", "Author Two"]
        assert paper.url == "https://arxiv.org/abs/2101.12345"

    def test_from_feed_entry(self, mock_feed_entry):
        paper = ArxivPaper.from_feed_entry(mock_feed_entry)
        
        assert paper.arxiv_id == "2101.12345"
        assert paper.title == "Test Paper"
        assert paper.abstract == "Test abstract"
        assert paper.authors == ["Author One", "Author Two"]
        assert paper.url == "https://arxiv.org/abs/2101.12345"

class TestHelperFunctions:
    def test_is_earlier(self):
        assert is_earlier("2101.12345", "2102.12345") is True
        assert is_earlier("2102.12345", "2101.12345") is False
        assert is_earlier("2101.12345", "2101.12345") is False

@pytest.fixture
def mock_config():
    class MockConfig:
        def __init__(self):
            self.FILTERING = {
                "max_results": "10",
                "force_primary": "True"
            }
            self.OUTPUT = {
                "debug_messages": "True"
            }
        
        def getboolean(self, key):
            return self.FILTERING[key] == "True"
    
    return MockConfig()

class TestPaperFetching:
    def test_get_papers_from_arxiv_api(self, mock_arxiv_client):
        timestamp = datetime.now()
        papers = get_papers_from_arxiv_api("cs.CL", timestamp, "2101.00000")
        
        assert len(papers) > 0
        assert all(isinstance(p, ArxivPaper) for p in papers)

    def test_get_papers_from_arxiv_rss(self, mock_config):
        papers, timestamp, last_id = get_papers_from_arxiv_rss("cs.CL", mock_config)
        
        assert isinstance(papers, list)
        assert all(isinstance(p, ArxivPaper) for p in papers)
        assert isinstance(timestamp, datetime)
        assert isinstance(last_id, str)

    def test_merge_paper_list(self):
        paper1 = ArxivPaper(arxiv_id="2101.12345")
        paper2 = ArxivPaper(arxiv_id="2102.12345")
        paper3 = ArxivPaper(arxiv_id="2101.12345")  # Duplicate
        
        merged = merge_paper_list([paper1], [paper2, paper3])
        
        assert len(merged) == 2
        assert merged[0].arxiv_id == "2102.12345"
        assert merged[1].arxiv_id == "2101.12345"