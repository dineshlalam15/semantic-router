"""Unit tests for ModelSelector and pluggable selection strategies."""

import pytest
from pathlib import Path

from app.config_loader import load_configurations
from app.model_selector import (
    ModelSelector,
    ModelSelectionError,
    SelectionStrategy,
    QualityOptimizedStrategy,
    CostOptimizedStrategy,
    LatencyOptimizedStrategy,
    PriorityStrategy,
    BalancedStrategy,
)
from app.schemas import ModelConfig


@pytest.fixture(scope="module")
def model_selector():
    """Initializes ModelSelector with models from config/models.yaml."""
    config = load_configurations(config_dir=Path("config"))
    return ModelSelector(
        models=config.models,
        default_strategy="quality",
        balanced_weights=config.settings.routing.balanced_weights,
    )


def test_candidate_filtering_for_domain(model_selector):
    """Verify that only models configured for the domain are returned as candidates."""
    candidates = model_selector.get_candidate_models("software_engineering")
    assert len(candidates) >= 3
    for m in candidates:
        assert "software_engineering" in m.domains


def test_unsupported_domain_raises_error(model_selector):
    """Verify ModelSelectionError when requesting a domain with no models."""
    with pytest.raises(ModelSelectionError):
        model_selector.select_model(domain="non_existent_domain")


def test_quality_strategy_selection(model_selector):
    """Quality strategy should prioritize high reasoning / coding capabilities."""
    model, strat, candidates = model_selector.select_model(
        domain="software_engineering", strategy_override="quality"
    )
    assert strat == "quality"
    # Claude 3.5 Sonnet has coding_capability 9.9 and reasoning 9.8
    assert model.model_name == "claude-3-5-sonnet-20241022"
    assert model.provider == "Anthropic"


def test_cost_strategy_selection(model_selector):
    """Cost strategy should select the cheapest candidate model."""
    model, strat, candidates = model_selector.select_model(
        domain="software_engineering", strategy_override="cost"
    )
    assert strat == "cost"
    # Qwen 2.5 Coder has input 0.30 + output 0.90 = 1.20, cheaper than Claude / GPT-4o
    assert model.model_name in ["qwen-2.5-coder-32b", "claude-3-5-haiku-20241022"]


def test_latency_strategy_selection(model_selector):
    """Latency strategy should select the lowest latency model."""
    model, strat, candidates = model_selector.select_model(
        domain="general_everyday_chat", strategy_override="latency"
    )
    assert strat == "latency"
    # Llama 3.1 8b has latency 190ms
    assert model.metadata.latency_p90_ms <= 350


def test_priority_strategy_selection(model_selector):
    """Priority strategy should pick model with highest configured priority integer."""
    model, strat, candidates = model_selector.select_model(
        domain="legal_document_analysis", strategy_override="priority"
    )
    assert strat == "priority"
    assert model.metadata.priority >= 9


def test_balanced_strategy_selection(model_selector):
    """Balanced strategy should return a valid candidate balancing attributes."""
    model, strat, candidates = model_selector.select_model(
        domain="stem_mathematics", strategy_override="balanced"
    )
    assert strat == "balanced"
    assert model in [c for c in model_selector.get_candidate_models("stem_mathematics")]


def test_extensible_custom_strategy(model_selector):
    """Test registering and executing a custom selection strategy dynamically."""
    class AlwaysProviderStrategy(SelectionStrategy):
        name = "custom_openai_only"

        def select(self, candidates, domain):
            for m in candidates:
                if m.provider == "OpenAI":
                    return m
            return candidates[0]

    model_selector.register_strategy(AlwaysProviderStrategy())
    model, strat, _ = model_selector.select_model(
        domain="software_engineering", strategy_override="custom_openai_only"
    )
    assert strat == "custom_openai_only"
    assert model.provider == "OpenAI"
