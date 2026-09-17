"""Unit tests for the HuggingFace encoder wrapper."""

import pytest
import numpy as np

from app.encoder import SemanticEncoder


@pytest.fixture(scope="module")
def encoder():
    """Initializes the SemanticEncoder once for testing."""
    return SemanticEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def test_encoder_dimension(encoder):
    """Verify embedding vector dimension is 384 for all-MiniLM-L6-v2."""
    assert encoder.dimension == 384


def test_encode_single(encoder):
    """Verify single query returns a normalized 1D vector of shape (384,)."""
    vec = encoder.encode_single("How do I implement binary search in Python?")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # L2 norm of unit vector should be approximately 1.0
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_encode_batch(encoder):
    """Verify batch query returns a normalized 2D matrix of shape (N, 384)."""
    texts = [
        "Calculate the eigenvalues of a matrix.",
        "Review this NDA contract for indemnification clauses.",
        "Write an engaging LinkedIn marketing hook.",
    ]
    mat = encoder.encode(texts)
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (3, 384)

    # Check each row is normalized to unit length
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_encode_empty_list(encoder):
    """Verify empty list returns an empty 2D array."""
    mat = encoder.encode([])
    assert mat.shape == (0, 384)
