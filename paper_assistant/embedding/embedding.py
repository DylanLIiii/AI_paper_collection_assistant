import json
import torch
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .paper import EmbeddingPaper
from loguru import logger


class PaperEmbedder:
    def __init__(
        self,
        model_name: str = "dunzhang/stella_en_400M_v5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        query_prompt: Optional[str] = "s2p_query",
    ):
        """
        Initialize the Stella embedder using sentence-transformers interface.

        Args:
            model_name: Name of the pre-trained model from Hugging Face
            device: Device to run the model on (cuda if available, else cpu)
            query_prompt: Prompt name for query encoding (s2p_query or s2s_query)
        """
        self.device = device
        self.query_prompt = query_prompt
        logger.info(f"Using device: {device}")
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, device=device
        )
        self.multi_process_pool = None

    def start_multi_process_pool(self) -> None:
        """
        Start a multi-process pool for parallel encoding across GPUs.
        Only available when using CUDA devices.
        """
        if self.device.startswith("cuda"):
            self.multi_process_pool = self.model.start_multi_process_pool()

    def stop_multi_process_pool(self) -> None:
        """
        Stop the multi-process pool and release resources.
        """
        if self.multi_process_pool is not None:
            self.model.stop_multi_process_pool(self.multi_process_pool)
            self.multi_process_pool = None

    def __del__(self):
        """
        Cleanup resources when the object is deleted.
        """
        self.stop_multi_process_pool()

    def embed_batch(
        self, papers: List[EmbeddingPaper], is_query: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of papers.

        Args:
            papers: List of EmbeddingPaper objects to embed
            is_query: Whether these are query embeddings (uses prompt if True)

        Returns:
            numpy.ndarray: Array of embeddings (num_papers x embedding_dim)
        """
        # Get text representation for each paper
        texts = [paper.get_text_for_embedding() for paper in papers]

        # Encode with appropriate prompt
        if is_query and self.query_prompt:
            if self.multi_process_pool is not None:
                embeddings = self.model.encode_multi_process(
                    sentences=texts,
                    pool=self.multi_process_pool,
                    prompt_name=self.query_prompt,
                    show_progress_bar=True,
                )
            else:
                embeddings = self.model.encode(
                    texts, prompt_name=self.query_prompt, convert_to_numpy=False
                )
        else:
            if self.multi_process_pool is not None:
                embeddings = self.model.encode_multi_process(
                    sentences=texts,
                    pool=self.multi_process_pool,
                    show_progress_bar=False,
                )
            else:
                embeddings = self.model.encode(texts, convert_to_numpy=True)

        return embeddings

    def embed_single(self, paper: EmbeddingPaper, is_query: bool = False) -> np.ndarray:
        """
        Generate embedding for a single paper.

        Args:
            paper: Single EmbeddingPaper object to embed
            is_query: Whether this is a query embedding (uses prompt if True)

        Returns:
            numpy.ndarray: Embedding vector (1 x embedding_dim)
        """
        return self.embed_batch([paper], is_query=is_query)[0]

    def similarity(
        self, query_embeddings: np.ndarray, doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity scores between query and document embeddings.

        Args:
            query_embeddings: Array of query embeddings
            doc_embeddings: Array of document embeddings

        Returns:
            numpy.ndarray: Similarity matrix (num_queries x num_docs)
        """
        return self.model.similarity(query_embeddings, doc_embeddings)


class EmbeddingProcessor:
    def __init__(self, output_path: str):
        """
        Initialize the embedding processor.

        Args:
            output_path: Path to save the output JSONL file
        """
        self.output_path = output_path
        self.file_handle = open(output_path, "w")

    def process_batch(self, papers: List[EmbeddingPaper], embeddings: np.ndarray):
        """
        Process a batch of papers with their embeddings and write to file.

        Args:
            papers: List of EmbeddingPaper objects
            embeddings: numpy array of embeddings (batch_size x embedding_dim)
        """
        for paper, embedding in zip(papers, embeddings):
            paper_dict = paper.to_dict()
            paper_dict["embedding"] = embedding.tolist()
            self.file_handle.write(json.dumps(paper_dict) + "\n")

    def close(self):
        """Close the output file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    paper_embedder = PaperEmbedder()
    paper = EmbeddingPaper(title="Test Paper", abstract="This is a test paper")
    embedding = paper_embedder.embed_single(paper)
    print(embedding)
    print(embedding.shape)
