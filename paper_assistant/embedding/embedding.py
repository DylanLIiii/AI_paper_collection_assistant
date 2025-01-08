from datetime import datetime
import json
import os
import torch
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from .paper import Paper
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

    def embed_batch(self, papers: List[Paper], is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for a batch of papers.

        Args:
            papers: List of Paper objects to embed
            is_query: Whether these are query embeddings (uses prompt if True)

        Returns:
            numpy.ndarray: Array of embeddings (num_papers x embedding_dim)
        """
        # Prepare texts
        texts = [self._prepare_text(paper) for paper in papers]

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

    def embed_single(self, paper: Paper, is_query: bool = False) -> np.ndarray:
        """
        Generate embedding for a single paper.

        Args:
            paper: Single Paper object to embed
            is_query: Whether this is a query embedding (uses prompt if True)

        Returns:
            numpy.ndarray: Embedding vector (1 x embedding_dim)
        """
        return self.embed_batch([paper], is_query=is_query)[0]

    def _prepare_text(self, paper: Paper) -> str:
        """
        Combine relevant paper fields into a single text string for embedding.
        """
        fields = ["title", "abstract"]
        text_parts = [str(getattr(paper, field, "")) for field in fields]
        return " ".join(text_parts)

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


# class PaperEmbedder:
#     def __init__(self, model_name: str = 'dunzhang/stella_en_400M_v5',
#                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
#         """
#         Initialize the text embedder with the stella model.

#         Args:
#             model_name: Name of the pre-trained model from Hugging Face
#             device: Device to run the model on (cuda or cpu)
#         """
#         self.device = device
#         self.model = AutoModel.from_pretrained(
#             model_name,
#             trust_remote_code=True
#         ).to(self.device).eval()

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             trust_remote_code=True
#         )

#         # Initialize linear layer for vector transformation
#         vector_dim = 1024
#         self.vector_linear = torch.nn.Linear(
#             in_features=self.model.config.hidden_size,
#             out_features=vector_dim
#         ).to(self.device)

#         # Load linear layer weights
#         vector_linear_directory = f"2_Dense_{vector_dim}"
#         vector_linear_dict = {
#             k.replace("linear.", ""): v for k, v in
#             torch.load(os.path.join(model_name, f"{vector_linear_directory}/pytorch_model.bin")).items()
#         }
#         self.vector_linear.load_state_dict(vector_linear_dict)

#     def _prepare_text(self, paper: Paper) -> str:
#         """
#         Combine relevant paper fields into a single text string for embedding.
#         """
#         fields = ['title', 'abstract']
#         text_parts = [str(getattr(paper, field, '')) for field in fields]
#         return ' '.join(text_parts)

#     def embed_batch(self, papers: List[Paper]) -> np.ndarray:
#         """
#         Generate embeddings for a batch of papers.

#         Args:
#             papers: List of Paper objects to embed

#         Returns:
#             numpy.ndarray: Array of embeddings (num_papers x embedding_dim)
#         """
#         # Prepare texts
#         texts = [self._prepare_text(paper) for paper in papers]

#         # Tokenize and move to device
#         with torch.no_grad():
#             input_data = self.tokenizer(
#                 texts,
#                 padding="longest",
#                 truncation=True,
#                 max_length=512,
#                 return_tensors="pt"
#             )
#             input_data = {k: v.to(self.device) for k, v in input_data.items()}

#             # Get model outputs
#             attention_mask = input_data["attention_mask"]
#             last_hidden_state = self.model(**input_data)[0]

#             # Mask and pool embeddings
#             last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
#             embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

#             # Apply linear transformation and normalize
#             embeddings = self.vector_linear(embeddings)
#             embeddings = normalize(embeddings.cpu().numpy())

#         return embeddings

#     def embed_single(self, paper: Paper) -> np.ndarray:
#         """
#         Generate embedding for a single paper.

#         Args:
#             paper: Single Paper object to embed

#         Returns:
#             numpy.ndarray: Embedding vector (1 x embedding_dim)
#         """
#         return self.embed_batch([paper])[0]


class EmbeddingProcessor:
    def __init__(self, output_path: str):
        """
        Initialize the embedding processor.

        Args:
            output_path: Path to save the output JSONL file
        """
        self.output_path = output_path
        self.file_handle = open(output_path, "w")

    def process_batch(self, papers: List[Paper], embeddings: np.ndarray):
        """
        Process a batch of papers with their embeddings and write to file.

        Args:
            papers: List of Paper objects
            embeddings: numpy array of embeddings (batch_size x embedding_dim)
        """
        for paper, embedding in zip(papers, embeddings):
            paper_dict = paper.to_dict()
            # handle datetime object
            for key, value in paper_dict.items():
                if isinstance(value, datetime):
                    paper_dict[key] = value.isoformat()
            paper_dict["embedding"] = embedding.tolist()  # Convert numpy array to list
            self.file_handle.write(json.dumps(paper_dict) + "\n")
        # Flush to ensure writing to disk
        self.file_handle.flush()

    def close(self):
        """
        Close the file handle.
        """
        if self.file_handle:
            self.file_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    paper_embedder = PaperEmbedder()
    paper = Paper(title="Test Paper", abstract="This is a test paper")
    embedding = paper_embedder.embed_single(paper)
    print(embedding)
    print(embedding.shape)
