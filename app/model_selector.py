"""Model selector with pluggable strategies for optimal model recommendation."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Type
import numpy as np

from app.schemas import ModelConfig, ModelCandidate, BalancedWeightsConfig

logger = logging.getLogger(__name__)


class ModelSelectionError(Exception):
    """Raised when no eligible models are found or selection fails."""
    pass


# ============================================================================
# Selection Strategy Pattern
# ============================================================================

class SelectionStrategy(ABC):
    """Abstract base class for all model selection strategies."""

    name: str = "base"

    @abstractmethod
    def select(self, candidates: List[ModelConfig], domain: str) -> ModelConfig:
        """Selects the optimal model from the candidate list for the given domain."""
        pass


class QualityOptimizedStrategy(SelectionStrategy):
    """Strategy prioritizing highest reasoning and domain capabilities."""

    name = "quality"

    def select(self, candidates: List[ModelConfig], domain: str) -> ModelConfig:
        if not candidates:
            raise ModelSelectionError(f"No candidates available to select from for domain '{domain}'.")

        def score_model(m: ModelConfig) -> float:
            reasoning = m.metadata.reasoning_capability or 5.0
            coding = m.metadata.coding_capability or 5.0
            priority = m.metadata.priority or 1

            if "software" in domain or "coding" in domain:
                quality_score = 0.6 * coding + 0.4 * reasoning
            else:
                quality_score = 0.7 * reasoning + 0.3 * coding

            # Incorporate priority as secondary tie-breaker
            return quality_score * 100 + priority

        return max(candidates, key=score_model)


class CostOptimizedStrategy(SelectionStrategy):
    """Strategy prioritizing lowest input + output API cost."""

    name = "cost"

    def select(self, candidates: List[ModelConfig], domain: str) -> ModelConfig:
        if not candidates:
            raise ModelSelectionError(f"No candidates available to select from for domain '{domain}'.")

        def cost_model(m: ModelConfig) -> float:
            in_cost = m.metadata.input_cost_per_m if m.metadata.input_cost_per_m is not None else 999.0
            out_cost = m.metadata.output_cost_per_m if m.metadata.output_cost_per_m is not None else 999.0
            # Blend 1:1 for basic estimation
            blended = in_cost + out_cost
            priority = m.metadata.priority or 1
            # Lower cost is better; tie-breaker prefers higher priority
            return blended - (priority * 0.001)

        return min(candidates, key=cost_model)


class LatencyOptimizedStrategy(SelectionStrategy):
    """Strategy prioritizing lowest P90 latency."""

    name = "latency"

    def select(self, candidates: List[ModelConfig], domain: str) -> ModelConfig:
        if not candidates:
            raise ModelSelectionError(f"No candidates available to select from for domain '{domain}'.")

        def latency_model(m: ModelConfig) -> float:
            lat = m.metadata.latency_p90_ms if m.metadata.latency_p90_ms is not None else 99999
            priority = m.metadata.priority or 1
            # Lower latency is better; tie-breaker prefers higher priority
            return lat - (priority * 0.1)

        return min(candidates, key=latency_model)


class PriorityStrategy(SelectionStrategy):
    """Strategy selecting by explicitly configured priority value."""

    name = "priority"

    def select(self, candidates: List[ModelConfig], domain: str) -> ModelConfig:
        if not candidates:
            raise ModelSelectionError(f"No candidates available to select from for domain '{domain}'.")

        return max(candidates, key=lambda m: (m.metadata.priority or 1, m.metadata.reasoning_capability or 0))


class BalancedStrategy(SelectionStrategy):
    """Strategy balancing quality, cost, and latency using normalized Pareto weighting."""

    name = "balanced"

    def __init__(self, weights: Optional[BalancedWeightsConfig] = None):
        self.weights = weights or BalancedWeightsConfig()

    def select(self, candidates: List[ModelConfig], domain: str) -> ModelConfig:
        if not candidates:
            raise ModelSelectionError(f"No candidates available to select from for domain '{domain}'.")
        if len(candidates) == 1:
            return candidates[0]

        # Extract metrics across candidates
        qualities = []
        costs = []
        latencies = []

        for m in candidates:
            reasoning = m.metadata.reasoning_capability or 5.0
            coding = m.metadata.coding_capability or 5.0
            q = (0.6 * coding + 0.4 * reasoning) if ("software" in domain or "coding" in domain) else (0.7 * reasoning + 0.3 * coding)
            c = (m.metadata.input_cost_per_m or 10.0) + (m.metadata.output_cost_per_m or 20.0)
            l = float(m.metadata.latency_p90_ms or 2000)

            qualities.append(q)
            costs.append(c)
            latencies.append(l)

        # Min-max normalization helper
        def normalize(vals, invert=False):
            arr = np.array(vals, dtype=float)
            min_v, max_v = arr.min(), arr.max()
            if max_v == min_v:
                return np.ones_like(arr) if not invert else np.ones_like(arr)
            norm = (arr - min_v) / (max_v - min_v)
            return (1.0 - norm) if invert else norm

        q_norm = normalize(qualities, invert=False)
        c_norm = normalize(costs, invert=True)       # Lower cost -> higher score
        l_norm = normalize(latencies, invert=True)   # Lower latency -> higher score

        w_q = self.weights.quality
        w_c = self.weights.cost
        w_l = self.weights.latency

        composite_scores = w_q * q_norm + w_c * c_norm + w_l * l_norm

        best_idx = int(np.argmax(composite_scores))
        return candidates[best_idx]


# ============================================================================
# Model Selector Engine
# ============================================================================

class ModelSelector:
    """Manages model candidate filtering and selection execution."""

    def __init__(
        self,
        models: List[ModelConfig],
        default_strategy: str = "quality",
        balanced_weights: Optional[BalancedWeightsConfig] = None,
    ):
        if not models:
            raise ValueError("ModelSelector requires at least one configured model.")
        self.models = models
        self.default_strategy_name = default_strategy.lower()

        # Register strategies
        self._strategies: Dict[str, SelectionStrategy] = {
            "quality": QualityOptimizedStrategy(),
            "cost": CostOptimizedStrategy(),
            "latency": LatencyOptimizedStrategy(),
            "priority": PriorityStrategy(),
            "balanced": BalancedStrategy(balanced_weights),
        }

        if self.default_strategy_name not in self._strategies:
            raise ValueError(f"Unknown default strategy: '{self.default_strategy_name}'")

    def register_strategy(self, strategy: SelectionStrategy):
        """Allows dynamic registration of custom selection strategies."""
        self._strategies[strategy.name.lower()] = strategy

    def get_candidate_models(self, domain: str) -> List[ModelConfig]:
        """Filters models that support the specified domain."""
        candidates = [m for m in self.models if domain in m.domains]
        return candidates

    def select_model(
        self, domain: str, strategy_override: Optional[str] = None
    ) -> (ModelConfig, str, List[ModelCandidate]):
        """Selects the optimal model for a domain given the active strategy.
        
        Returns:
            Tuple of (selected_model, active_strategy_name, candidate_models_list)
        """
        candidates = self.get_candidate_models(domain)
        if not candidates:
            # Fallback: if no direct domain match, inspect capability match or log error
            raise ModelSelectionError(
                f"No configured models found for domain '{domain}'. "
                f"Please update models.yaml to assign models to this domain."
            )

        strat_name = (strategy_override or self.default_strategy_name).lower()
        strategy = self._strategies.get(strat_name)
        if not strategy:
            logger.warning(f"Unknown strategy '{strat_name}', falling back to default '{self.default_strategy_name}'")
            strat_name = self.default_strategy_name
            strategy = self._strategies[strat_name]

        selected_model = strategy.select(candidates, domain)

        # Build candidate representations for observability
        candidate_summary = [
            ModelCandidate(
                provider=m.provider,
                model_name=m.model_name,
                priority=m.metadata.priority,
                input_cost_per_m=m.metadata.input_cost_per_m,
                output_cost_per_m=m.metadata.output_cost_per_m,
                latency_p90_ms=m.metadata.latency_p90_ms,
                reasoning_capability=m.metadata.reasoning_capability,
            )
            for m in candidates
        ]

        return selected_model, strat_name, candidate_summary
