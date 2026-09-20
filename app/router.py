"""Semantic Router: encodes queries with HuggingFace and selects models from models.yaml."""

import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import yaml
from semantic_router.encoders import HuggingFaceEncoder
from app.schemas import RouteResponse

logger = logging.getLogger(__name__)


class SemanticRouter:
    """Classifies queries into domains and maps them to LLM models."""

    def __init__(
        self,
        routes_path: str = "config/routes.yaml",
        text_models_path: str = "config/text_models.yaml",
        image_models_path: str = "config/image_models.yaml",
    ):
        self.routes_path = Path(routes_path)
        self.text_models_path = Path(text_models_path)
        self.image_models_path = Path(image_models_path)

        # 1. Load routes from YAML
        with open(self.routes_path, "r", encoding="utf-8") as f:
            routes_data = yaml.safe_load(f) or {}
        self.routes: List[Dict[str, Any]] = routes_data.get("routes", [])

        # 2. Load and merge text + image models at startup
        with open(self.text_models_path, "r", encoding="utf-8") as f:
            text_models_data = yaml.safe_load(f) or {}
        with open(self.image_models_path, "r", encoding="utf-8") as f:
            image_models_data = yaml.safe_load(f) or {}
        self.models: List[Dict[str, Any]] = (
            text_models_data.get("models", []) + image_models_data.get("models", [])
        )

        # 3. Initialize Hugging Face encoder
        logger.info(
            "Initializing HuggingFaceEncoder | text_models=%s image_models=%s",
            self.text_models_path,
            self.image_models_path,
        )
        self.encoder = HuggingFaceEncoder()

        # 4. Flatten utterances and map them to their domain names
        self._utterances: List[str] = []
        self._domains: List[str] = []
        for r in self.routes:
            domain_name = r["name"]
            for utt in r.get("utterances", []):
                self._utterances.append(utt)
                self._domains.append(domain_name)

        # 5. Pre-compute normalized embeddings for all utterances at startup
        logger.info(f"Pre-computing embeddings for {len(self._utterances)} utterances...")
        self._matrix = self._encode_normalized(self._utterances)

    def _encode_normalized(self, texts: List[str]) -> np.ndarray:
        """
        Encodes texts into L2-normalized vectors.
            - Input: batch of strings (texts)
            - Returns: 2D NumPy array of normalized vectors (unit vectors)
        """ 
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        raw = self.encoder(texts)
        arr = np.array(raw, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    def route(self, query: str) -> RouteResponse:
        """Finds the best matching domain and returns the recommended model."""
        # Embed query and find best matching domain via cosine similarity
        query_vec = self._encode_normalized([query])[0]
        similarities = np.dot(self._matrix, query_vec)
        best_idx = int(np.argmax(similarities))
        domain = self._domains[best_idx]

        # Lookup model for the domain in models.yaml
        model = next((m for m in self.models if domain in m.get("domains", [])), None)
        if not model:
            raise ValueError(f"No configured model found for domain '{domain}'")

        return RouteResponse(
            query=query,
            domain=domain,
            recommended_model=model["model_name"],
            llm_provider=model["provider"],
        )
