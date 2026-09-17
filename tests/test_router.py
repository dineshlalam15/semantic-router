"""Unit tests for semantic domain router and deterministic ranking."""

import pytest
from pathlib import Path

from app.config_loader import load_configurations
from app.encoder import SemanticEncoder
from app.semantic_router import SemanticDomainRouter


@pytest.fixture(scope="module")
def domain_router():
    """Initializes SemanticDomainRouter once for tests."""
    config = load_configurations(config_dir=Path("config"))
    encoder = SemanticEncoder(model_name=config.settings.encoder.model_name, device="cpu")
    return SemanticDomainRouter(routes=config.routes, encoder=encoder)


def test_software_engineering_routing(domain_router):
    """Test queries related to coding route to software_engineering."""
    query = "How do I implement an LRU cache in Python?"
    res = domain_router.route(query)
    assert res.selected_domain == "software_engineering"
    assert res.confidence_score > 0.6
    assert len(res.ranked_candidates) == 5
    assert res.ranked_candidates[0].domain == "software_engineering"


def test_stem_mathematics_routing(domain_router):
    """Test mathematical and physics queries route to stem_mathematics."""
    query = "Calculate the derivative of f(x) = x^3 * sin(x) with respect to x."
    res = domain_router.route(query)
    assert res.selected_domain == "stem_mathematics"
    assert res.confidence_score > 0.6


def test_legal_analysis_routing(domain_router):
    """Test legal and compliance queries route to legal_document_analysis."""
    query = "What are the data privacy compliance requirements under GDPR Article 6?"
    res = domain_router.route(query)
    assert res.selected_domain == "legal_document_analysis"
    assert res.confidence_score > 0.6


def test_content_marketing_routing(domain_router):
    """Test copywriting and marketing queries route to content_marketing."""
    query = "Write 5 high-converting email newsletter subject lines for a new SaaS feature launch."
    res = domain_router.route(query)
    assert res.selected_domain == "content_marketing"
    assert res.confidence_score > 0.6


def test_general_chat_routing(domain_router):
    """Test casual dialogue routes to general_everyday_chat."""
    query = "Good morning! Can you suggest a quick and healthy breakfast idea?"
    res = domain_router.route(query)
    assert res.selected_domain == "general_everyday_chat"
    assert res.confidence_score > 0.5


def test_ambiguous_query_handling(domain_router):
    """Test ambiguous queries that touch multiple domains."""
    query = "Build a machine learning neural network to forecast quantitative equity market returns."
    res = domain_router.route(query)
    # Both software_engineering and stem_mathematics are strong candidates
    assert res.selected_domain in ["software_engineering", "stem_mathematics"]
    assert len(res.ranked_candidates) == 5
    # Verify candidate scores are sorted in descending order
    scores = [c.similarity_score for c in res.ranked_candidates]
    assert scores == sorted(scores, reverse=True)


def test_no_threshold_dropout(domain_router):
    """Verify router never drops queries or returns None even for off-distribution queries."""
    query = "What is the airspeed velocity of an unladen European swallow?"
    res = domain_router.route(query)
    assert res.selected_domain is not None
    assert isinstance(res.selected_domain, str)
    assert res.confidence_score > 0.0
