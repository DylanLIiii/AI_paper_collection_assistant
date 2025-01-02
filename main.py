import json
import configparser
import os
from datetime import datetime
from typing import Generator, List

import litellm
from openai import OpenAI
from requests import Session
import io

from litellm import completion
import instructor
import arxiv

from arxiv_scraper import get_papers_from_arxiv_rss_api, Paper, EnhancedJSONEncoder
from filter_papers import filter_by_author, filter_by_gpt
from parse_json_to_md import render_md_string
from push_to_slack import push_to_slack
from helpers import get_api_key
from utils import batched, argsort, get_batch, get_one_author

class PaperProcessor:
    def __init__(self, config):
        self.config = config
        self.S2_API_KEY = None

    def get_papers_from_arxiv(self) -> List[Paper]:
        """Fetch papers from arXiv based on configured categories."""
        area_list = self.config["FILTERING"]["arxiv_category"].split(",")
        paper_set = set()
        for area in area_list:
            papers = get_papers_from_arxiv_rss_api(area.strip(), self.config)
            paper_set.update(set(papers))
        if self.config["OUTPUT"].getboolean("debug_messages"):
            print(f"Number of papers: {len(paper_set)}")
        return list(paper_set)

    def get_papers(self, ids: List[str], batch_size: int = 100, **kwargs) -> Generator[dict, None, None]:
        """Get papers from Semantic Scholar API using batching."""
        with Session() as session:
            for ids_batch in batched(ids, batch_size=batch_size):
                yield from get_batch(session, ids_batch, self.S2_API_KEY, "paper", **kwargs)

    def get_authors(self, all_authors: List[str], batch_size: int = 100, **kwargs):
        """Get author metadata from Semantic Scholar API."""
        debug_file = "/Users/dylanli/repos/gpt_paper_assistant/out/all_authors.debug.json"
        if os.path.exists(debug_file):
            with open(debug_file, 'r') as f:
                return json.load(f)
                
        author_metadata_dict = {}
        with Session() as session:
            for author in tqdm(all_authors):
                auth_map = get_one_author(session, author, self.S2_API_KEY)
                if auth_map is not None:
                    author_metadata_dict[author] = auth_map
                time.sleep(0.02 if self.S2_API_KEY else 1.0)
        return author_metadata_dict

    def parse_authors(self, lines):
        """Parse the comma-separated author list."""
        author_ids = []
        authors = []
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            author_split = line.split(",")
            author_ids.append(author_split[1].strip())
            authors.append(author_split[0].strip())
        return authors, author_ids

class QAPaperProcessor:
    def __init__(self, client, config):
        self.client = client
        self.config = config

    def ask_questions_for_paper(self, paper: Paper, questions: List[str]) -> dict:
        """Ask questions about a paper using an AI model."""
        try:
            search = arxiv.Search(id_list=[paper.arxiv_id])
            results = list(arxiv.Client().results(search))
            if not results:
                return {"error": "PDF not found"}

            paper_entry = next(results)
            pdf_filename = f"out/pdfs/{paper.arxiv_id}.pdf"
            paper_entry.download_pdf(filename=pdf_filename)

            md = MarkItDown()
            result = md.convert(pdf_filename)
            text_content = result.text_content

            question_answers = {}
            for question in questions:
                qa_context = "\n\n".join([
                    f"Q: {q}\nA: {a}" for q, a in question_answers.items()
                ])
                
                prompt = f"""Paper Content:
                            {text_content[:50000]}

                            Previous Questions and Answers:
                            {qa_context}

                            Current Question: {question}

                            Please answer the current question, taking into account the previous Q&A if relevant."""

                response = self.client.chat.completions.create(
                    model=self.config["SELECTION"]["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_retries=3,
                    timeout=10,
                )
                question_answers[question] = response.choices[0].message.content
            return question_answers
        except Exception as e:
            print(f"Error processing paper {paper.arxiv_id}: {e}")
            return {"error": str(e)}

    def generate_qa_markdown(self, paper: Paper, question_answers: dict):
        """Generate a markdown file with questions and answers for a paper."""
        os.makedirs("out/papers", exist_ok=True)
        filepath = f"out/papers/{paper.arxiv_id}.md"
        with open(filepath, "w") as f:
            f.write(f"# {paper.title}\n\n")
            for question, answer in question_answers.items():
                f.write(f"**Q:** {question}\n\n")
                f.write(f"**A:** {answer}\n\n")

def main():
    try:
        api_key = get_api_key()
        os.environ["GEMINI_API_KEY"] = api_key
        
        config = configparser.ConfigParser()
        config.read("configs/config.ini")

        client = instructor.from_litellm(completion)
        paper_processor = PaperProcessor(config)
        qa_processor = QAPaperProcessor(client, config)

        # Load author list
        with io.open("configs/authors.txt", "r") as fopen:
            author_names, author_ids = paper_processor.parse_authors(fopen.readlines())
        author_id_set = set(author_ids)

        # Process papers
        papers = paper_processor.get_papers_from_arxiv()
        all_authors = set(author for paper in papers for author in paper.authors)
        
        if config["OUTPUT"].getboolean("debug_messages"):
            print(f"Getting author info for {len(all_authors)} authors")
        
        author_metadata = paper_processor.get_authors(list(all_authors))

        # Filter and process papers
        selected_papers, all_papers, sort_dict = filter_by_author(
            author_metadata, papers, author_id_set, config
        )
        
        filter_by_gpt(
            author_metadata,
            papers,
            config,
            client,
            all_papers,
            selected_papers,
            sort_dict,
        )

        # Sort and output results
        sorted_keys = [keys[idx] for idx in argsort(list(sort_dict.values()))[::-1]]
        selected_papers = {key: selected_papers[key] for key in sorted_keys}

        if config["OUTPUT"].getboolean("debug_messages"):
            print(sort_dict)
            print(selected_papers)

        # Output results
        if papers:
            if config["OUTPUT"].getboolean("dump_json"):
                with open(config["OUTPUT"]["output_path"] + "output.json", "w") as outfile:
                    json.dump(selected_papers, outfile, indent=4)
            
            if config["OUTPUT"].getboolean("dump_md"):
                today = datetime.now().strftime("%Y-%m-%d")
                formatted_papers = {
                    key: Paper(**paper_dict) if isinstance(paper_dict, dict) else paper_dict
                    for key, paper_dict in selected_papers.items()
                }
                with open(config["OUTPUT"]["output_path"] + f"{today}_output.md", "w") as f:
                    f.write(render_md_string(formatted_papers))
            
            if config["OUTPUT"].getboolean("push_to_slack"):
                SLACK_KEY = os.environ.get("SLACK_KEY")
                if SLACK_KEY:
                    push_to_slack(selected_papers)
                else:
                    print("Warning: push_to_slack is true, but SLACK_KEY is not set")

    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
