"""
Shared test fixtures and path configuration.
All tests import from backend/ by adding it to sys.path here.
"""
import sys
import os

# Ensure backend/ is importable regardless of how pytest is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Reusable SearchResults fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_result():
    """One matching document with full metadata."""
    return SearchResults(
        documents=["Python is a high-level programming language."],
        metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
        distances=[0.12],
    )


@pytest.fixture
def multi_result():
    """Multiple matching documents across different lessons."""
    return SearchResults(
        documents=[
            "Functions are reusable blocks of code.",
            "Classes define object blueprints.",
        ],
        metadata=[
            {"course_title": "Python Basics", "lesson_number": 2},
            {"course_title": "Python Basics", "lesson_number": 3},
        ],
        distances=[0.15, 0.22],
    )


@pytest.fixture
def no_lesson_result():
    """Document whose metadata does not include a lesson number."""
    return SearchResults(
        documents=["General course overview content."],
        metadata=[{"course_title": "Intro Course"}],  # no lesson_number key
        distances=[0.30],
    )


@pytest.fixture
def empty_result():
    """Empty result set — no matching documents."""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_result():
    """Result set that carries an error message from the store."""
    return SearchResults(
        documents=[], metadata=[], distances=[], error="ChromaDB connection failed"
    )


# ---------------------------------------------------------------------------
# Mock VectorStore
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """Bare MagicMock standing in for VectorStore."""
    return MagicMock()
