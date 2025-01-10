import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from .embedding import PaperEmbedder
from .paper import Paper
from loguru import logger
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker:
    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None
    ):
        """
        Initialize the reranker model.

        Args:
            model_name: Name of the reranker model
            device: Device to use (cuda/cpu). If None, will auto-detect.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing reranker on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents based on their relevance to the query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None returns all)

        Returns:
            List of (document, score) tuples sorted by relevance
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            scores = (
                self.model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )

        # Pair documents with their scores and sort
        scored_docs = list(zip(documents, scores.cpu().numpy().tolist()))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k] if top_k is not None else scored_docs


class SemanticSearch:
    def __init__(self, embeddings_file: str, use_reranker: bool = False):
        """
        Initialize semantic search with pre-computed embeddings.

        Args:
            embeddings_file: Path to the JSONL file containing papers with embeddings
            use_reranker: Whether to use reranker for final ranking
        """
        # Initialize device first
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Initialize embedder with the same device
        self.embedder = PaperEmbedder(device=self.device)

        # Initialize storage
        self.papers: List[Dict] = []
        self.embeddings: List[np.ndarray] = []

        # Initialize reranker if requested
        self.reranker = Reranker() if use_reranker else None

        # Load embeddings last
        self._load_embeddings(embeddings_file)

    def _load_embeddings(self, embeddings_file: str) -> None:
        """Load papers and their embeddings from the JSONL file."""
        logger.info(f"Loading embeddings from {embeddings_file}")
        with open(embeddings_file, "r") as f:
            for line in f:
                paper_dict = json.loads(line.strip())
                embedding = np.array(paper_dict.pop("embedding"), dtype=np.float32)
                self.papers.append(paper_dict)
                self.embeddings.append(embedding)

        self.embeddings = np.vstack(self.embeddings).astype(np.float32)
        # Convert embeddings to torch tensor and move to appropriate device
        self.embeddings = torch.from_numpy(self.embeddings).to(self.device)
        logger.info(f"Loaded {len(self.papers)} papers with embeddings")

    def search(
        self, query: str, top_k: int = 5, rerank_top_k: Optional[int] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Search for papers similar to the query.

        Args:
            query: Search query text
            top_k: Number of top results to return from initial search
            rerank_top_k: Number of top results to rerank (None to rerank all top_k results)

        Returns:
            List of tuples containing (paper_dict, similarity_score)
        """
        # Create a Paper object for the query
        query_paper = Paper(title=query, abstract="")

        # Get query embedding and ensure it's on the same device and dtype
        query_embedding = self.embedder.embed_single(query_paper, is_query=True)
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.astype(np.float32)
            query_embedding = torch.from_numpy(query_embedding)
        query_embedding = query_embedding.to(self.device).float()  # Ensure float32

        # Calculate similarities
        similarities = self.embedder.similarity(
            query_embedding.reshape(1, -1), self.embeddings
        )

        # Move similarities to CPU for numpy operations
        similarities = similarities[0].cpu().numpy()

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Get initial results
        initial_results = []
        for idx in top_indices:
            initial_results.append((self.papers[idx], float(similarities[idx])))

        # Apply reranking if enabled
        if self.reranker is not None:
            # Extract texts for reranking
            texts_to_rerank = [
                paper["title"] + " " + paper["abstract"]
                for paper, _ in initial_results[: rerank_top_k or top_k]
            ]

            # Rerank the texts
            reranked = self.reranker.rerank(
                query=query, documents=texts_to_rerank, top_k=top_k
            )

            # Map reranked results back to original papers
            reranked_results = []
            for text, score in reranked:
                # Find matching paper
                for paper, _ in initial_results:
                    if (paper["title"] + " " + paper["abstract"]) == text:
                        reranked_results.append((paper, score))
                        break
            return reranked_results

        return initial_results


if __name__ == "__main__":
    # Example usage
    embeddings_file = "path/to/your/embeddings.jsonl"  # Update this path
    searcher = SemanticSearch(embeddings_file)

    # Example queries
    test_queries = [
        "Large language models and their applications",
        "Computer vision object detection",
        "Natural language processing transformers",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        results = searcher.search(query, top_k=3)

        for paper, score in results:
            print(f"\nTitle: {paper['title']}")
            print(f"Score: {score:.4f}")
            print(f"Abstract: {paper['abstract'][:200]}...")
