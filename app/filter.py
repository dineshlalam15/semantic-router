"""
Metadata-based model filtering and selection module.

Evaluates candidate models within a domain against query-level constraints
(latency SLAs, context window limits, budget caps, and privacy/on-premise requirements)
using model metadata and capabilities.
"""

import logging
from typing import List, Dict, Any, Optional

from app.pattern import (
    _extract_max_latency_ms,
    _extract_min_context_k,
    _requires_on_premise,
    _requires_low_cost,
    _extract_required_capabilities,
)

logger = logging.getLogger(__name__)


def _rank_by_metadata_utility(
    candidates: List[Dict[str, Any]],
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """Ranks surviving candidate models based on metadata efficiency (latency, cost) and capability match."""
    if not candidates:
        raise ValueError("Cannot rank empty candidates list")
    if len(candidates) == 1:
        return candidates[0]

    q_lower = (query or "").lower()

    def score_model(idx: int, model: Dict[str, Any]) -> float:
        meta = model.get("metadata", {}) or {}
        caps = model.get("capabilities", []) or []

        # 1. Base score from YAML priority order (first model is designated champion)
        score = 1.0 / (idx + 1.0)

        # 2. Capability matching bonus
        for cap in caps:
            cap_norm = cap.replace("_", " ").lower()
            if any(term in q_lower for term in cap_norm.split()):
                score += 0.3

        # 3. Latency efficiency bonus (lower latency = higher score)
        latency = meta.get("avg_latency_ms")
        if latency:
            # Latency between 300ms and 5000ms mapped to bonus 0.0 -> 0.4
            score += max(0.0, (5000.0 - latency) / 10000.0)

        # 4. Cost efficiency bonus (lower token/image cost = higher score)
        cost_in = meta.get("cost_per_1k_input_tokens")
        cost_img = meta.get("cost_per_image")
        cost = cost_in if cost_in is not None else cost_img
        if cost is not None:
            score += max(0.0, (0.01 - min(cost, 0.01)) * 20.0)

        return score

    scored = [(score_model(idx, m), m) for idx, m in enumerate(candidates)]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_model = scored[0]

    logger.debug(
        "Ranked %d models. Winner: %s/%s (score=%.3f)",
        len(candidates),
        best_model.get("provider"),
        best_model.get("model_name"),
        best_score,
    )
    return best_model


def _filter_by_metadata(
    available_models: List[Dict[str, Any]],
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """Filters available candidate models by query constraints and metadata, returning the optimal model.

    Args:
        available_models: List of model configuration dictionaries matching the query's domain.
        query: Optional user prompt string to extract operational constraints (latency, context, budget, privacy).

    Returns:
        The single optimal model configuration dictionary.
    """
    if not available_models:
        raise ValueError("available_models cannot be empty")

    if len(available_models) == 1:
        return available_models[0]

    candidates = list(available_models)

    if not query:
        return _rank_by_metadata_utility(candidates)

    # --- Check 1: Latency SLA Check ---
    max_latency = _extract_max_latency_ms(query)
    if max_latency is not None:
        latency_filtered = [
            m for m in candidates
            if m.get("metadata", {}).get("avg_latency_ms") is not None
            and m["metadata"]["avg_latency_ms"] <= max_latency
        ]
        if latency_filtered:
            logger.info("Filtered by latency <= %dms: %d/%d models passed", max_latency, len(latency_filtered), len(candidates))
            candidates = latency_filtered
        else:
            logger.warning("No model satisfied latency <= %dms; keeping candidates for relaxed scoring", max_latency)

    # --- Check 2: Context Window / Document Size Check ---
    min_context_k = _extract_min_context_k(query)
    if min_context_k is not None:
        context_filtered = [
            m for m in candidates
            if m.get("metadata", {}).get("context_window_k") is not None
            and m["metadata"]["context_window_k"] >= min_context_k
        ]
        if context_filtered:
            logger.info("Filtered by context >= %dk: %d/%d models passed", min_context_k, len(context_filtered), len(candidates))
            candidates = context_filtered

    # --- Check 3: Data Governance / On-Premise / Open-Weights Check ---
    if _requires_on_premise(query):
        privacy_filtered = [
            m for m in candidates
            if any("open_weights" in cap.lower() for cap in m.get("capabilities", []))
        ]
        if privacy_filtered:
            logger.info("Filtered by on-premise/open-weights: %d/%d models passed", len(privacy_filtered), len(candidates))
            candidates = privacy_filtered

    # --- Check 4: Budget & Low-Cost Check ---
    if _requires_low_cost(query):
        # Sort candidates prioritizing lowest cost per unit
        candidates.sort(
            key=lambda m: (
                m.get("metadata", {}).get("cost_per_1k_input_tokens") or
                m.get("metadata", {}).get("cost_per_image") or
                999.0
            )
        )

    # --- Check 5: Domain Specific Capabilities Check ---
    req_caps = _extract_required_capabilities(query)
    if req_caps:
        cap_filtered = [
            m for m in candidates
            if any(
                any(req in cap.lower() for req in req_caps)
                for cap in m.get("capabilities", [])
            )
        ]
        if cap_filtered:
            candidates = cap_filtered

    # Rank surviving candidates via multi-objective utility scoring
    return _rank_by_metadata_utility(candidates, query=query)


# Alias without leading underscore for public export
filter_by_metadata = _filter_by_metadata
