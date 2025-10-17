import logging
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridCacheManager:
    """
    Cache manager to reduce redundant retrievals.
    Stores embeddings + chunks for fast similarity lookups.
    """

    def __init__(self, model: str = "sentence-transformers/paraphrase-MiniLM-L6-v2", max_cache_size: int = 500):
        self.encoder = SentenceTransformer(model)
        self.cache: Dict[str, Tuple[np.ndarray, str]] = {}  # {chunk_id: (embedding, text)}
        self.max_cache_size = max_cache_size

    def embed_text(self, text: str) -> np.ndarray:
        """Utility: get embedding from SentenceTransformer"""
        return self.encoder.encode([text])[0]

    def add_chunks(self, chunks: List[str]) -> None:
        """
        Add new chunks to cache (with embeddings).
        If max_cache_size exceeded → evict oldest.
        """
        for chunk in chunks:
            chunk_id = str(hash(chunk))  # simple unique ID
            if chunk_id not in self.cache:
                if len(self.cache) >= self.max_cache_size:
                    self.evict_chunks(1)
                embedding = self.embed_text(chunk)
                self.cache[chunk_id] = (embedding, chunk)
                logger.info(f"Added chunk to cache: {chunk[:50]}...")

    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve most relevant cached chunks for query.
        Uses cosine similarity between query and cached embeddings.
        """
        if not self.cache:
            return []

        query_vec = self.embed_text(query)
        similarities = []
        for chunk_id, (embedding, text) in self.cache.items():
            sim = cosine_similarity([query_vec], [embedding])[0][0]
            similarities.append((sim, text))

        similarities.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [text for _, text in similarities[:top_k]]

        logger.info(f"Retrieved {len(top_chunks)} cached chunks for query: {query}")
        return top_chunks

    def evict_chunks(self, n: int = 1) -> None:
        """
        Evict n oldest chunks from cache (FIFO).
        """
        if not self.cache:
            return
        evicted_keys = list(self.cache.keys())[:n]
        for key in evicted_keys:
            del self.cache[key]
            logger.info(f"Evicted chunk {key} from cache.")
