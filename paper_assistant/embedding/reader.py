import json
import os
from typing import Generator, Callable
from tqdm import tqdm
from .paper import Paper
import random
from loguru import logger
from typing import List
from io import StringIO


class PaperReader:
    def __init__(self, file_path: str):
        """
        Initialize the PaperReader with the path to a large JSON file.
        """
        self.file_path = file_path

    def stream_papers(self) -> Generator[Paper, None, None]:
        """
        Stream Paper objects one by one from the JSON file.
        Each line in the file is a complete JSON object.

        Returns:
            Generator[Paper, None, None]: A generator that yields Paper objects.
        """
        with open(self.file_path, "r") as file:
            for line in file:
                try:
                    item = json.loads(line.strip())
                    yield Paper(**item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")

    def filter(
        self, paper_filter: Callable[[Paper], bool], limit: int = None
    ) -> Generator[Paper, None, None]:
        """
        Filter papers using a PaperFilter or other callable condition.

        Args:
            paper_filter: A PaperFilter or callable that takes a Paper object and returns True if it matches
            limit: Maximum number of papers to yield. Defaults to None.

        Returns:
            Generator[Paper, None, None]: A generator that yields filtered Paper objects.
        """
        count = 0
        for paper in tqdm(self.stream_papers(), desc="Filtering papers", unit="paper"):
            if paper_filter(paper):
                yield paper
                count += 1
                if limit is not None and count >= limit:
                    break

    def save_to_json(
        self,
        output_path: str,
        papers: Generator[Paper, None, None] = None,
        paper_filter: Callable[[Paper], bool] = None,
        limit: int = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save papers to a JSONL (JSON Lines) file, optionally filtered and limited.

        Args:
            output_path: Path to save the JSONL file
            papers: Optional generator of Paper objects to save
            paper_filter: Optional PaperFilter or callable to filter papers
            limit: Maximum number of papers to save
            overwrite: Whether to overwrite existing file

        Raises:
            ValueError: If invalid parameters are provided
            IOError: If file operations fail
        """
        # Validate parameters
        if not output_path:
            raise ValueError("output_path must be provided")
        if papers is not None and paper_filter is not None:
            raise ValueError("Cannot specify both papers generator and paper_filter")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        if os.path.exists(output_path) and not overwrite:
            logger.warning(f"File {output_path} already exists and overwrite=False")
            return

        try:
            count = 0
            # Determine paper source
            paper_source = (
                papers
                if papers is not None
                else self.filter(paper_filter, limit)
                if paper_filter is not None
                else self.stream_papers()
            )

            with open(output_path, "w") as output_file:
                buffer = StringIO()
                buffer_size = 5000

                for paper in paper_source:
                    buffer.write(json.dumps(paper.to_dict()) + "\n")
                    count += 1

                    if count % buffer_size == 0:
                        output_file.write(buffer.getvalue())
                        buffer.truncate(0)
                        buffer.seek(0)

                    if limit is not None and count >= limit:
                        break

                # Write remaining buffer content
                if buffer.tell() > 0:
                    output_file.write(buffer.getvalue())

            logger.info(
                f"Successfully saved {count} papers to {output_path} in JSONL format"
            )

        except IOError as e:
            logger.error(f"Error writing to file {output_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while saving papers: {e}")
            raise

    def stream_batches(
        self, batch_size: int = 32
    ) -> Generator[List[Paper], None, None]:
        """
        Stream papers in batches of specified size.

        Args:
            batch_size: Number of papers per batch

        Returns:
            Generator[List[Paper], None, None]: A generator that yields lists of Paper objects
        """
        batch = []
        for paper in self.stream_papers():
            batch.append(paper)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # Yield any remaining papers
            yield batch

    def sample(self, n: int = 1, buffer_size: int = 10) -> list[Paper]:
        """
        Randomly sample n papers from the dataset using reservoir sampling.
        This method is memory-efficient as it doesn't load the entire dataset into memory.

        Args:
            n (int): Number of papers to sample. Defaults to 1.
            buffer_size (int): Size of the buffer for reservoir sampling. Defaults to 10000.

        Returns:
            list[Paper]: List of randomly sampled Paper objects
        """
        if n < 1:
            raise ValueError("Sample size must be at least 1")

        # Initialize reservoir with first n papers
        reservoir = []
        stream = self.stream_papers()

        # Fill the reservoir with first n items
        for _ in range(n):
            try:
                reservoir.append(next(stream))
            except StopIteration:
                return reservoir  # Return all papers if dataset has fewer than n items

        # Process remaining items with reservoir sampling
        for i, paper in enumerate(stream, start=n):
            j = random.randint(0, i)
            if j < n:
                reservoir[j] = paper

            # Optional: break after processing buffer_size papers
            if i >= buffer_size:
                break

        return reservoir

    def count_papers(self):
        # This function should be used for counting the number of papers. Only use for a small number of papers.
        # So we need to check the file size and then use the stream_papers function to count the number of papers.
        file_size = os.path.getsize(self.file_path)
        if file_size < 1000000000:
            count = 0
            for _ in self.stream_papers():
                count += 1
            return count
        else:
            logger.error(
                "File size is too large to count papers. Returning None. Please use the stream_papers function instead."
            )
            return None


if __name__ == "__main__":
    reader = PaperReader(
        "/datadrive2/hengl/data/arxiv/212/arxiv-metadata-oai-snapshot.json"
    )
    sampled_papers = reader.sample(10)
    for paper in sampled_papers:
        print(paper.to_dict())
        print("-" * 80)
