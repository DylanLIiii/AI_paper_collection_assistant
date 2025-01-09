import configparser
from typing import Dict, List
from paper_assistant.core.arxiv_scraper import ArxivPaper as Paper
from paper_assistant.core.arxiv_scraper import get_papers_from_arxiv_api
from paper_assistant.utils.cache_handler import CacheHandler
from litellm import completion
import instructor
from pydantic import BaseModel
import os


class QaResult(BaseModel):
    question: str
    answer: str

class TableQaResult(BaseModel):
    results: str

class QaProcessor:
    # Define standard columns for the table
    STANDARD_COLUMNS = [
        "arxiv_id",
        "title",
        "abstract",
        "key_contributions",
        "methodology",
        "results",
        "limitations",
        "practical_applications"
    ]
    
    def __init__(self, api_key=None):
        # Load config
        self.config = configparser.ConfigParser()
        self.config.read("configs/config.ini")

        # Set API key in environment if provided
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

        # Initialize client
        self.client = instructor.from_litellm(completion)

        # Load questions
        with open("configs/questions.txt", "r") as f:
            self.questions = [line.strip() for line in f.readlines() if line.strip()]

        # Progress tracking
        self.progress = {}


        self.cache_handler = CacheHandler("out/qa_cache")

    def generate_table_row(self, paper: Paper) -> Dict[str, str]:
        """Generate a table row for a single paper"""
        # Get basic info
        row = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "abstract": paper.abstract
        }
        
        # Get PDF content
        text_content = paper.pdf_content
        if not text_content:
            text_content = paper.abstract

        # Generate additional columns using LLM
        for column in self.STANDARD_COLUMNS[3:]:  # Skip first 3 standard columns
            prompt = f"""Paper Content:
{text_content[:50000]}

Extract the {column.replace('_', ' ')} from this paper. 
- Be concise but informative
- Use markdown formatting:
  * Use **bold** for important terms
  * Use *italics* for emphasis
  * Use bullet points (-) for lists
  * Use `code` for technical terms
  * Use $...$ for inline math and $$...$$ for block math
  * Use \n for line breaks
- Focus on key information
"""
            try:
                response = self.client.chat.completions.create(
                    model=self.config["SELECTION"]["model"],
                    response_model=TableQaResult,
                    messages=[{"role": "user", "content": prompt}],
                    max_retries=3,
                    timeout=30,
                )
                row[column] = response.results
            except Exception as e:
                row[column] = f"Error: {str(e)}"
        
        return row

    def generate_paper_table(self, papers: List[Paper]) -> List[Dict[str, str]]:
        """Generate table data from multiple papers"""
        table_data = []
        for paper in papers:
            row = self.generate_table_row(paper)
            table_data.append(row)
        return table_data

    def process_qa(self, paper: Paper, progress_callback=None) -> Dict[str, str]:
        """Process Q&A for a paper with caching"""
        try:
            paper_id = paper.arxiv_id

            # Check cache first
            cached_results = self.cache_handler.get_cached_data(paper_id)
            if cached_results:
                print(f"Using cached Q&A for paper {paper_id}")
                return cached_results

            # Initialize progress
            self.progress[paper_id] = {"current": 0, "total": len(self.questions)}

            # Get paper content
            text_content = paper.pdf_content
            if not text_content:
                text_content = paper.abstract

            # Process each question
            qa_results = {}

            # rules
            base_rules = """
            You are a helpful assistant that answers questions about a paper.
            You are given a paper and a question.
            You are to answer the question based on the paper.
            - list as bullet points with markdown formatting.
            - contain important details for each bullet point.
            """

            for i, question in enumerate(self.questions, 1):
                try:
                    # Update progress
                    self.progress[paper_id]["current"] = i
                    if progress_callback:
                        progress_callback(paper_id, i, len(self.questions))

                    # Include previous Q&A pairs in the context
                    qa_context = "\n\n".join(
                        [f"Q: {q}\nA: {a}" for q, a in qa_results.items()]
                    )

                    # Normal questions use the standard format
                    prompt = f"""Paper Content:
                                {text_content[:50000]}

                                    Previous Questions and Answers:
                                    {qa_context}

                                    Current Question: {question}

                                    Rules:
                                    {base_rules}

                                    Please answer the current question, taking into account the previous Q&A if relevant."""

                    response = self.client.chat.completions.create(
                        model=self.config["SELECTION"]["model"],
                        response_model=QaResult,
                        messages=[{"role": "user", "content": prompt}],
                        max_retries=3,
                        timeout=30,
                    )
                    qa_results[question] = response.answer

                except Exception as e:
                    qa_results[question] = f"Error getting answer: {str(e)}"

            # Save results to cache
            self.cache_handler.save_cache_data(paper_id, qa_results)

            return qa_results

        except Exception as e:
            print(f"Error processing Q&A for paper {paper.arxiv_id}: {e}")
            return {"error": str(e)}
        finally:
            if paper.arxiv_id in self.progress:
                del self.progress[paper.arxiv_id]

    def get_progress(self, paper_id: str) -> Dict[str, int]:
        """Get current progress for a paper"""
        return self.progress.get(paper_id, {"current": 0, "total": 0})

if __name__ == "__main__":
    # Create test papers
    keys_config = configparser.ConfigParser()
    keys_config.read("configs/keys.ini")

    api_key = keys_config["GEMINI"]["api_key"]
    test_papers = get_papers_from_arxiv_api(arxiv_ids=["2310.16834", "2310.16779"])

    # Initialize processor
    qa_processor = QaProcessor(api_key)

    from paper_assistant.utils.table_parser import TableParser
    
    # Generate table data
    table_data = qa_processor.generate_paper_table(test_papers)
    
    # Initialize parser
    parser = TableParser(table_data)
    
    # Generate markdown
    markdown_table = parser.to_markdown()
    print("Generated Markdown Table:")
    print(markdown_table)
    
    # Save outputs
    with open("test_table.md", "w") as f:
        f.write("# Test Paper Table\n\n")
        f.write(markdown_table)
    
    # Save as Excel
    parser.to_excel("test_table.xlsx")
    
    # Generate HTML
    parser.to_html("test_table.html", title="Test Paper Summaries")
    
    print("\nSaved outputs to:")
    print("- test_table.md")
    print("- test_table.xlsx") 
    print("- test_table.html")
