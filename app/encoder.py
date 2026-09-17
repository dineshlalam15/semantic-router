"""HuggingFace encoder wrapper using semantic_router.encoders.HuggingFaceEncoder."""

import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)


class EncoderInitializationError(Exception):
    """Raised when HuggingFaceEncoder fails to initialize or download weights."""
    pass


class SemanticEncoder:
    """Wrapper around semantic_router.encoders.HuggingFaceEncoder.
    
    Provides normalized vector outputs for efficient linear algebraic cosine similarity.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._encoder = None
        self._dimension = None
        self._initialize_encoder()

    def _initialize_encoder(self):
        """Initializes the underlying HuggingFaceEncoder."""
        try:
            logger.info(f"Loading HuggingFaceEncoder model: '{self.model_name}' on device: '{self.device}'...")
            from semantic_router.encoders import HuggingFaceEncoder

            # HuggingFaceEncoder takes name=model_name
            self._encoder = HuggingFaceEncoder(name=self.model_name)
            
            # Warm up and determine embedding dimension
            test_vec = self._encoder(["warmup"])
            test_arr = np.array(test_vec, dtype=np.float32)
            self._dimension = test_arr.shape[1]
            logger.info(f"HuggingFaceEncoder loaded successfully. Embedding dimension: {self._dimension}")
        except Exception as exc:
            logger.error(f"Failed to initialize HuggingFaceEncoder with model '{self.model_name}': {exc}", exc_info=True)
            raise EncoderInitializationError(
                f"Could not load HuggingFaceEncoder ('{self.model_name}'). "
                f"Ensure PyTorch and Transformers are installed: {exc}"
            ) from exc

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimension."""
        return self._dimension

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of texts into an L2-normalized 2D float32 NumPy matrix (N, D).
        
        L2 normalization ensures that cosine similarity is equivalent to dot product.
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        raw_embeddings = self._encoder(texts)
        arr = np.array(raw_embeddings, dtype=np.float32)

        # L2-normalize vectors along axis 1
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        normalized = arr / norms
        return normalized

    def encode_single(self, text: str) -> np.ndarray:
        """Encodes a single query string into an L2-normalized 1D float32 NumPy array (D,)."""
        mat = self.encode([text])
        return mat[0]
