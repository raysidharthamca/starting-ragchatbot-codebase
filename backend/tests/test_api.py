"""
Tests for the FastAPI API endpoints (/api/query, /api/courses).

Uses an inline test app (instead of importing backend/app.py directly) to avoid
the StaticFiles mount and live RAGSystem initialization that run at module load
time in the real app.  The endpoint logic mirrors app.py exactly so the tests
exercise the same request/response contract.
"""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import Any, List, Optional
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Shared mock — replaced per-test via the reset_rag fixture below
# ---------------------------------------------------------------------------

_rag = MagicMock()


# ---------------------------------------------------------------------------
# Inline test app — same endpoint logic as app.py, no static files
# ---------------------------------------------------------------------------

test_app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Any]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


@test_app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        session_id = request.session_id
        if not session_id:
            session_id = _rag.session_manager.create_session()
        answer, sources = _rag.query(request.query, session_id)
        return QueryResponse(answer=answer, sources=sources, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@test_app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    try:
        analytics = _rag.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_rag():
    """Reset the shared RAG mock and configure sensible defaults before each test."""
    _rag.reset_mock()
    _rag.session_manager.create_session.return_value = "test-session-001"
    _rag.query.return_value = ("Default answer", [])
    _rag.get_course_analytics.return_value = {
        "total_courses": 0,
        "course_titles": [],
    }
    yield _rag


@pytest.fixture
def client():
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_returns_200_on_valid_request(self, client):
        resp = client.post("/api/query", json={"query": "What is Python?"})
        assert resp.status_code == 200

    def test_response_contains_answer(self, client):
        _rag.query.return_value = ("Python is a programming language.", [])
        resp = client.post("/api/query", json={"query": "What is Python?"})
        assert resp.json()["answer"] == "Python is a programming language."

    def test_response_contains_session_id(self, client):
        resp = client.post("/api/query", json={"query": "q"})
        data = resp.json()
        assert "session_id" in data
        assert data["session_id"] != ""

    def test_auto_creates_session_when_none_provided(self, client):
        resp = client.post("/api/query", json={"query": "q"})
        assert resp.json()["session_id"] == "test-session-001"
        _rag.session_manager.create_session.assert_called_once()

    def test_uses_provided_session_id(self, client):
        resp = client.post("/api/query", json={"query": "q", "session_id": "existing-session"})
        assert resp.json()["session_id"] == "existing-session"
        _rag.session_manager.create_session.assert_not_called()

    def test_passes_query_to_rag_system(self, client):
        client.post("/api/query", json={"query": "What is MCP?"})
        _rag.query.assert_called_once()
        positional_query = _rag.query.call_args[0][0]
        assert positional_query == "What is MCP?"

    def test_passes_session_id_to_rag_query(self, client):
        client.post("/api/query", json={"query": "q", "session_id": "sess-xyz"})
        call_args = _rag.query.call_args[0]
        assert call_args[1] == "sess-xyz"

    def test_returns_sources_in_response(self, client):
        sources = [{"label": "Python Basics - Lesson 1", "url": "http://example.com"}]
        _rag.query.return_value = ("Answer", sources)
        resp = client.post("/api/query", json={"query": "q"})
        assert resp.json()["sources"] == sources

    def test_empty_sources_returned_as_list(self, client):
        _rag.query.return_value = ("Answer", [])
        resp = client.post("/api/query", json={"query": "q"})
        assert resp.json()["sources"] == []

    def test_missing_query_field_returns_422(self, client):
        resp = client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_rag_exception_returns_500(self, client):
        _rag.query.side_effect = Exception("AI service unavailable")
        resp = client.post("/api/query", json={"query": "q"})
        assert resp.status_code == 500

    def test_500_detail_contains_error_message(self, client):
        _rag.query.side_effect = Exception("AI service unavailable")
        resp = client.post("/api/query", json={"query": "q"})
        assert "AI service unavailable" in resp.json()["detail"]

    def test_session_creation_error_returns_500(self, client):
        _rag.session_manager.create_session.side_effect = RuntimeError("Session store full")
        resp = client.post("/api/query", json={"query": "q"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/api/courses")
        assert resp.status_code == 200

    def test_returns_total_courses_count(self, client):
        _rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["A", "B", "C"],
        }
        resp = client.get("/api/courses")
        assert resp.json()["total_courses"] == 3

    def test_returns_course_titles_list(self, client):
        _rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Python Basics", "Advanced FastAPI"],
        }
        resp = client.get("/api/courses")
        assert resp.json()["course_titles"] == ["Python Basics", "Advanced FastAPI"]

    def test_empty_catalog_returns_zero_and_empty_list(self, client):
        resp = client.get("/api/courses")
        data = resp.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_total_courses_matches_titles_length(self, client):
        _rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course A", "Course B"],
        }
        resp = client.get("/api/courses")
        data = resp.json()
        assert data["total_courses"] == len(data["course_titles"])

    def test_rag_exception_returns_500(self, client):
        _rag.get_course_analytics.side_effect = Exception("DB connection lost")
        resp = client.get("/api/courses")
        assert resp.status_code == 500

    def test_500_detail_contains_error_message(self, client):
        _rag.get_course_analytics.side_effect = Exception("DB connection lost")
        resp = client.get("/api/courses")
        assert "DB connection lost" in resp.json()["detail"]
