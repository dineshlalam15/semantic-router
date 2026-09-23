"""
Capability & Metadata efficiency filtering module.

Selects the optimal model from domain-matched candidate models using:
1. Dynamic semantic capability matching (threshold >= 0.35).
2. Operational metadata efficiency (cost & latency) fallback when no specific capability is demanded.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

"""
A model capability must have a cosine similarity >=0.35 with the query embedding to be considered a genuine match.
"""
CAPABILITY_THRESHOLD = 0.35


def _score_metadata_efficiency(model: Dict[str, Any]) -> float:
    """
    This function evaluates how cheap and fast a model is when the user query
    doesn't require any specialized capability. 
    """

    meta = model.get("metadata", {}) or {}
    cost_in = meta.get("cost_per_1k_input_tokens")
    cost_img = meta.get("cost_per_image")
    cost = cost_in if cost_in is not None else cost_img
    cost_score = 1.0 / (1.0 + (cost * 100.0)) if cost is not None else 0.5

    latency = meta.get("avg_latency_ms")
    latency_score = 5000.0 / (5000.0 + latency) if latency is not None else 0.5

    return (cost_score * 0.6) + (latency_score * 0.4)


def _filter_capabilities(
    candidates: List[Dict[str, Any]],
    capability_scores: Optional[Dict[str, float]] = None,
    threshold: float = CAPABILITY_THRESHOLD,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Selects the optimal candidate model.
    1. Capability Match: Model with the highest cumulative similarity for capabilities >= threshold.
    2. Metadata Fallback: If no model meets the threshold, pick the cheapest and fastest model.
    """
    if not candidates:
        raise ValueError("Candidate models list cannot be empty")
    if len(candidates) == 1:
        return candidates[0]

    # 1. Capability Matching (pick model with highest cumulative score)
    best_model = None
    best_score = 0.0

    if capability_scores:
        for model in candidates:
            # Sum scores of capabilities that meet or exceed the threshold
            total_score = sum(
                capability_scores.get(cap, 0.0)
                for cap in model.get("capabilities", [])
                if capability_scores.get(cap, 0.0) >= threshold
            )
            if total_score > best_score:
                best_score = total_score
                best_model = model

        if best_model:
            logger.info(
                "Selected via capability match: %s/%s (cumulative_score=%.3f)",
                best_model.get("provider"),
                best_model.get("model_name"),
                best_score,
            )
            return best_model

    # 2. Metadata Efficiency Fallback (cheapest & fastest)
    winner = max(candidates, key=_score_metadata_efficiency)
    logger.info(
        "No capability match >= %.2f. Selected via metadata efficiency: %s/%s",
        threshold,
        winner.get("provider"),
        winner.get("model_name"),
    )
    return winner

