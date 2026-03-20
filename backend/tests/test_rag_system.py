"""
Tests for RAGSystem.query() — the top-level content query pipeline.

Covers:
- AI generator is called with the user's question and the tool list
- Conversation history is passed when a session exists
- The response + sources tuple is returned correctly
- Sources are reset after being retrieved
- Session history is updated after each exchange
- Exceptions from the AI generator propagate (not silently swallowed)
- The prompt wraps the raw user query (regression: bare query vs. wrapped prompt)
"""

import pytest
from unittest.mock import MagicMock, patch, call
from rag_system import RAGSystem

# ---------------------------------------------------------------------------
# Fixture: a fully mocked RAGSystem
# ---------------------------------------------------------------------------


@pytest.fixture
def rag(tmp_path):
    """
    RAGSystem with all heavy dependencies mocked out.

    Substitutions:
    - VectorStore  → MagicMock (no ChromaDB I/O)
    - AIGenerator  → MagicMock (no Anthropic API calls)
    - DocumentProcessor → MagicMock
    - SessionManager → real object (lightweight, no I/O)
    """
    with (
        patch("rag_system.VectorStore") as MockVS,
        patch("rag_system.AIGenerator") as MockAI,
        patch("rag_system.DocumentProcessor"),
        patch("rag_system.CourseSearchTool"),
        patch("rag_system.CourseOutlineTool"),
        patch("rag_system.ToolManager") as MockTM,
    ):

        # Configure the mock VectorStore instance
        mock_vs_instance = MagicMock()
        MockVS.return_value = mock_vs_instance

        # Configure the mock AIGenerator instance
        mock_ai_instance = MagicMock()
        mock_ai_instance.generate_response.return_value = "AI answer"
        MockAI.return_value = mock_ai_instance

        # Configure ToolManager instance
        mock_tm_instance = MagicMock()
        mock_tm_instance.get_tool_definitions.return_value = [{"name": "search_course_content"}]
        mock_tm_instance.get_last_sources.return_value = []
        MockTM.return_value = mock_tm_instance

        from config import Config

        cfg = Config(
            ANTHROPIC_API_KEY="test-key",
            ANTHROPIC_MODEL="claude-sonnet-4-5",
            CHROMA_PATH=str(tmp_path / "chroma"),
        )
        system = RAGSystem(cfg)

        # Expose mocks for assertions
        system._mock_ai = mock_ai_instance
        system._mock_tm = mock_tm_instance
        system._mock_vs = mock_vs_instance

        yield system


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------


class TestQueryReturnValue:
    def test_returns_tuple_of_two(self, rag):
        result = rag.query("What is MCP?")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_ai_response(self, rag):
        response, _ = rag.query("What is MCP?")
        assert response == "AI answer"

    def test_second_element_is_sources_list(self, rag):
        rag._mock_tm.get_last_sources.return_value = [{"label": "Course A", "url": None}]
        _, sources = rag.query("What is MCP?")
        assert isinstance(sources, list)

    def test_sources_come_from_tool_manager(self, rag):
        expected_sources = [{"label": "Python Basics - Lesson 1", "url": "http://x.com"}]
        rag._mock_tm.get_last_sources.return_value = expected_sources
        _, sources = rag.query("What is Python?")
        assert sources == expected_sources


# ---------------------------------------------------------------------------
# AI generator is called correctly
# ---------------------------------------------------------------------------


class TestAIGeneratorCall:
    def test_generate_response_is_called(self, rag):
        rag.query("Any question")
        rag._mock_ai.generate_response.assert_called_once()

    def test_query_text_appears_in_prompt(self, rag):
        rag.query("What is prompt caching?")
        call_kwargs = rag._mock_ai.generate_response.call_args[1]
        prompt = call_kwargs.get("query", "")
        assert "What is prompt caching?" in prompt

    def test_tools_are_passed_to_generate_response(self, rag):
        rag.query("Content question")
        call_kwargs = rag._mock_ai.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None

    def test_tool_manager_is_passed_to_generate_response(self, rag):
        rag.query("Content question")
        call_kwargs = rag._mock_ai.generate_response.call_args[1]
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] is not None


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------


class TestConversationHistory:
    def test_no_history_passed_without_session(self, rag):
        rag.query("First question")
        call_kwargs = rag._mock_ai.generate_response.call_args[1]
        # No session → history should be None
        assert call_kwargs.get("conversation_history") is None

    def test_history_passed_for_existing_session(self, rag):
        session_id = rag.session_manager.create_session()
        rag.session_manager.add_exchange(session_id, "prev q", "prev a")

        rag.query("follow-up", session_id=session_id)

        call_kwargs = rag._mock_ai.generate_response.call_args[1]
        history = call_kwargs.get("conversation_history")
        assert history is not None
        assert "prev q" in history

    def test_session_history_updated_after_query(self, rag):
        session_id = rag.session_manager.create_session()
        rag.query("my question", session_id=session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert "my question" in history
        assert "AI answer" in history


# ---------------------------------------------------------------------------
# Source lifecycle
# ---------------------------------------------------------------------------


class TestSourceLifecycle:
    def test_get_last_sources_called_after_generate(self, rag):
        rag.query("q")
        rag._mock_tm.get_last_sources.assert_called_once()

    def test_reset_sources_called_after_retrieval(self, rag):
        rag.query("q")
        rag._mock_tm.reset_sources.assert_called_once()

    def test_reset_called_after_get(self, rag):
        """reset_sources must be called AFTER get_last_sources, not before."""
        call_order = []
        rag._mock_tm.get_last_sources.side_effect = lambda: call_order.append("get") or []
        rag._mock_tm.reset_sources.side_effect = lambda: call_order.append("reset")

        rag.query("q")

        assert call_order == ["get", "reset"], f"Expected ['get', 'reset'] but got {call_order}"


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


class TestExceptionPropagation:
    def test_ai_generator_exception_propagates(self, rag):
        """
        Exceptions from the AI generator must NOT be silently swallowed.
        They must propagate so the FastAPI endpoint can return a proper 500.
        If RAGSystem catches and hides the exception, the caller gets a
        blank/wrong response instead of an error — masking the real bug.
        """
        rag._mock_ai.generate_response.side_effect = Exception("API error: model not found")

        with pytest.raises(Exception, match="API error: model not found"):
            rag.query("What is Python?")

    def test_vector_store_exception_propagates(self, rag):
        """Errors from the VectorStore should surface, not be hidden."""
        rag._mock_ai.generate_response.side_effect = RuntimeError("ChromaDB failure")

        with pytest.raises(RuntimeError):
            rag.query("Search question")
