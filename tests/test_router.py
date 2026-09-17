"""Unit tests for SemanticRouter domain matching and model selection."""

import pytest
from app.router import SemanticRouter


@pytest.fixture(scope="module")
def router():
    """Initializes SemanticRouter once for tests."""
    return SemanticRouter()


def test_software_engineering_routing(router):
    """Test queries related to coding route to software_engineering."""
    res = router.route("How do I implement an LRU cache in Python?")
    assert res.domain == "software_engineering"
    assert res.recommended_model == "claude-3-5-sonnet-20241022"
    assert res.llm_provider == "Anthropic"


def test_stem_mathematics_routing(router):
    """Test mathematical queries route to stem_mathematics."""
    res = router.route("Calculate the derivative of f(x) = x^3 * sin(x) with respect to x.")
    assert res.domain == "stem_mathematics"
    assert res.recommended_model in ["o1", "deepseek-r1"]


def test_legal_analysis_routing(router):
    """Test legal and compliance queries route to legal_document_analysis."""
    res = router.route("What are the data privacy compliance requirements under GDPR Article 6?")
    assert res.domain == "legal_document_analysis"


def test_content_marketing_routing(router):
    """Test marketing queries route to content_marketing."""
    res = router.route("Write 5 high-converting email newsletter subject lines for a new SaaS feature launch.")
    assert res.domain == "content_marketing"


def test_general_chat_routing(router):
    """Test casual dialogue routes to general_everyday_chat."""
    res = router.route("Good morning! Can you suggest a quick and healthy breakfast idea?")
    assert res.domain == "general_everyday_chat"


def test_image_generation_routing(router):
    """Test text-to-image prompts route to image_generation."""
    res = router.route("Generate a photorealistic portrait of an astronaut on Mars in watercolor style.")
    assert res.domain == "image_generation"
    assert res.recommended_model == "dall-e-3"
    assert res.llm_provider == "OpenAI"


def test_no_threshold_dropout(router):
    """Verify router always assigns the nearest domain even for off-distribution queries."""
    res = router.route("What is the airspeed velocity of an unladen European swallow?")
    assert res.domain is not None
    assert res.recommended_model is not None
    assert res.llm_provider is not None
