"""Semantic router engine with pre-computed embeddings and deterministic non-threshold ranking."""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from app.schemas import RouteConfig, DomainCandidate
from app.encoder import SemanticEncoder

logger = logging.getLogger(__name__)


@dataclass
class DomainRoutingResult:
    """Detailed result of semantic domain routing."""
    selected_domain: str
    confidence_score: float
    matched_utterance: str
    ranked_candidates: List[DomainCandidate]
    is_ambiguous: bool = False
    ambiguity_margin: float = 0.0


class SemanticDomainRouter:
    """Performs semantic similarity matching against configured domain utterances.
    
    Adheres strictly to the non-threshold ranking design:
    1. Pre-computes all utterance embeddings into a single normalized matrix U at startup.
    2. Given a query embedding q, computes cosine similarities via dot product: s = U @ q.
    3. Aggregates scores per domain by finding the maximum utterance match for that domain.
    4. Ranks all candidate domains deterministically by score (with route priority as tie-breaker).
    5. Selects the highest-ranking domain without rejecting or dropping queries via manual thresholds.
    """

    def __init__(self, routes: List[RouteConfig], encoder: SemanticEncoder):
        if not routes:
            raise ValueError("SemanticDomainRouter requires at least one configured route.")
        self.routes = routes
        self.encoder = encoder

        self._utterance_texts: List[str] = []
        self._utterance_domains: List[str] = []
        self._domain_priorities: Dict[str, int] = {r.name: r.priority for r in routes}
        self._domain_to_indices: Dict[str, List[int]] = {r.name: [] for r in routes}
        self._utterance_matrix: Optional[np.ndarray] = None

        self._build_index()

    def _build_index(self):
        """Pre-computes and caches normalized embeddings for all route utterances at startup."""
        logger.info("Pre-computing route utterance embeddings matrix at application startup...")
        idx = 0
        all_texts = []
        for route in self.routes:
            for utterance in route.utterances:
                all_texts.append(utterance)
                self._utterance_texts.append(utterance)
                self._utterance_domains.append(route.name)
                self._domain_to_indices[route.name].append(idx)
                idx += 1

        self._utterance_matrix = self.encoder.encode(all_texts)
        logger.info(
            f"Pre-computed embeddings for {len(all_texts)} utterances across {len(self.routes)} domains. "
            f"Matrix shape: {self._utterance_matrix.shape}"
        )

    def route(self, query: str) -> DomainRoutingResult:
        """Deterministically classifies a user query into the best-matching domain.
        
        Args:
            query: The user query text.
            
        Returns:
            DomainRoutingResult with the selected domain, score, and full candidate ranking.
        """
        # 1. Encode query to unit vector (shape: D)
        query_vec = self.encoder.encode_single(query)

        # 2. Vectorized cosine similarity dot product: (N, D) @ (D,) -> (N,)
        similarities = np.dot(self._utterance_matrix, query_vec)

        # 3. Aggregate by domain
        domain_scores: List[Tuple[str, float, str, int]] = []
        for route in self.routes:
            indices = self._domain_to_indices[route.name]
            if not indices:
                continue
            domain_sims = similarities[indices]
            best_local_idx = int(np.argmax(domain_sims))
            best_score = float(domain_sims[best_local_idx])
            best_global_idx = indices[best_local_idx]
            best_utterance = self._utterance_texts[best_global_idx]
            priority = self._domain_priorities.get(route.name, 1)

            domain_scores.append((route.name, best_score, best_utterance, priority))

        # 4. Deterministic Ranking: Sort primarily by score desc, secondarily by priority desc
        # Using a stable sort with key: (score, priority)
        domain_scores.sort(key=lambda x: (round(x[1], 5), x[3]), reverse=True)

        top_domain, top_score, top_utterance, _ = domain_scores[0]

        # 5. Check for ambiguity (margin between top-1 and top-2)
        is_ambiguous = False
        ambiguity_margin = 0.0
        if len(domain_scores) > 1:
            second_score = domain_scores[1][1]
            ambiguity_margin = top_score - second_score
            if ambiguity_margin < 0.05:
                is_ambiguous = True
                logger.info(
                    f"Ambiguous query detected: Top='{top_domain}' ({top_score:.3f}), "
                    f"Runner-up='{domain_scores[1][0]}' ({second_score:.3f}), margin={ambiguity_margin:.3f}"
                )

        # 6. Format ranked candidates
        ranked_candidates = [
            DomainCandidate(
                domain=item[0],
                similarity_score=round(item[1], 4),
                top_matched_utterance=item[2]
            )
            for item in domain_scores
        ]

        return DomainRoutingResult(
            selected_domain=top_domain,
            confidence_score=round(top_score, 4),
            matched_utterance=top_utterance,
            ranked_candidates=ranked_candidates,
            is_ambiguous=is_ambiguous,
            ambiguity_margin=round(ambiguity_margin, 4)
        )
