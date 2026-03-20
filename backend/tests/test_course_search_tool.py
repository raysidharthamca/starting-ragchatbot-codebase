"""
Tests for CourseSearchTool.execute() and ToolManager integration.

Covers:
- Output format when results are returned
- Error propagation from the vector store
- Empty-result messages (with and without filters)
- Parameter forwarding to VectorStore.search()
- Source tracking (label + URL)
- Tool definition structure
"""
import pytest
from unittest.mock import MagicMock, call
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


# ===========================================================================
# Helpers
# ===========================================================================

def make_tool(store):
    return CourseSearchTool(store)


# ===========================================================================
# Output formatting
# ===========================================================================

class TestExecuteFormatting:
    def test_result_contains_course_title(self, mock_store, single_result):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = None

        tool = make_tool(mock_store)
        output = tool.execute(query="Python basics")

        assert "Python Basics" in output

    def test_result_contains_lesson_number(self, mock_store, single_result):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = None

        tool = make_tool(mock_store)
        output = tool.execute(query="Python basics")

        assert "Lesson 1" in output

    def test_result_contains_document_text(self, mock_store, single_result):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = None

        tool = make_tool(mock_store)
        output = tool.execute(query="Python basics")

        assert "Python is a high-level programming language." in output

    def test_multiple_results_all_included(self, mock_store, multi_result):
        mock_store.search.return_value = multi_result
        mock_store.get_lesson_link.return_value = None

        tool = make_tool(mock_store)
        output = tool.execute(query="functions and classes")

        assert "Functions are reusable blocks of code." in output
        assert "Classes define object blueprints." in output
        assert "Lesson 2" in output
        assert "Lesson 3" in output

    def test_result_without_lesson_number_omits_lesson_label(
        self, mock_store, no_lesson_result
    ):
        mock_store.search.return_value = no_lesson_result
        mock_store.get_lesson_link.return_value = None

        tool = make_tool(mock_store)
        output = tool.execute(query="overview")

        assert "Intro Course" in output
        # "Lesson None" must NOT appear
        assert "Lesson None" not in output


# ===========================================================================
# Error and empty paths
# ===========================================================================

class TestExecuteErrorAndEmpty:
    def test_store_error_is_returned_verbatim(self, mock_store, error_result):
        mock_store.search.return_value = error_result

        tool = make_tool(mock_store)
        output = tool.execute(query="anything")

        assert "ChromaDB connection failed" in output

    def test_empty_results_message_contains_no_relevant_content(
        self, mock_store, empty_result
    ):
        mock_store.search.return_value = empty_result

        tool = make_tool(mock_store)
        output = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in output

    def test_empty_with_course_filter_mentions_course_name(
        self, mock_store, empty_result
    ):
        mock_store.search.return_value = empty_result

        tool = make_tool(mock_store)
        output = tool.execute(query="topic", course_name="Python Basics")

        assert "Python Basics" in output

    def test_empty_with_lesson_filter_mentions_lesson_number(
        self, mock_store, empty_result
    ):
        mock_store.search.return_value = empty_result

        tool = make_tool(mock_store)
        output = tool.execute(query="topic", lesson_number=3)

        assert "3" in output


# ===========================================================================
# Parameter forwarding
# ===========================================================================

class TestExecuteParameterForwarding:
    def test_search_called_with_query(self, mock_store, single_result):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = None

        tool = make_tool(mock_store)
        tool.execute(query="test query")

        mock_store.search.assert_called_once()
        args, kwargs = mock_store.search.call_args
        assert kwargs.get("query") == "test query" or args[0] == "test query"

    def test_search_called_with_course_name(self, mock_store, single_result):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = None

        tool = make_tool(mock_store)
        tool.execute(query="content", course_name="Python Basics")

        _, kwargs = mock_store.search.call_args
        assert kwargs.get("course_name") == "Python Basics"

    def test_search_called_with_lesson_number(self, mock_store, single_result):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = None

        tool = make_tool(mock_store)
        tool.execute(query="content", lesson_number=5)

        _, kwargs = mock_store.search.call_args
        assert kwargs.get("lesson_number") == 5

    def test_search_defaults_course_name_to_none(self, mock_store, empty_result):
        mock_store.search.return_value = empty_result

        tool = make_tool(mock_store)
        tool.execute(query="anything")

        _, kwargs = mock_store.search.call_args
        assert kwargs.get("course_name") is None

    def test_search_defaults_lesson_number_to_none(self, mock_store, empty_result):
        mock_store.search.return_value = empty_result

        tool = make_tool(mock_store)
        tool.execute(query="anything")

        _, kwargs = mock_store.search.call_args
        assert kwargs.get("lesson_number") is None


# ===========================================================================
# Source tracking
# ===========================================================================

class TestSourceTracking:
    def test_sources_populated_after_execute(self, mock_store, single_result):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = make_tool(mock_store)
        tool.execute(query="test")

        assert len(tool.last_sources) == 1

    def test_source_label_contains_course_and_lesson(self, mock_store, single_result):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = None

        tool = make_tool(mock_store)
        tool.execute(query="test")

        label = tool.last_sources[0]["label"]
        assert "Python Basics" in label
        assert "1" in label

    def test_source_url_populated_when_link_available(self, mock_store, single_result):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = make_tool(mock_store)
        tool.execute(query="test")

        assert tool.last_sources[0]["url"] == "https://example.com/lesson1"

    def test_source_url_is_none_when_no_link(self, mock_store, single_result):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = None

        tool = make_tool(mock_store)
        tool.execute(query="test")

        assert tool.last_sources[0]["url"] is None

    def test_source_url_none_when_no_lesson_number(self, mock_store, no_lesson_result):
        """get_lesson_link must NOT be called when there is no lesson_number."""
        mock_store.search.return_value = no_lesson_result

        tool = make_tool(mock_store)
        tool.execute(query="overview")

        # url should be None and get_lesson_link should not have been called
        assert tool.last_sources[0]["url"] is None
        mock_store.get_lesson_link.assert_not_called()

    def test_multiple_results_produce_multiple_sources(self, mock_store, multi_result):
        mock_store.search.return_value = multi_result
        mock_store.get_lesson_link.return_value = None

        tool = make_tool(mock_store)
        tool.execute(query="test")

        assert len(tool.last_sources) == 2

    def test_sources_empty_on_no_results(self, mock_store, empty_result):
        mock_store.search.return_value = empty_result

        tool = make_tool(mock_store)
        tool.execute(query="test")

        assert tool.last_sources == []

    def test_sources_reset_between_calls(self, mock_store, single_result, empty_result):
        mock_store.search.side_effect = [single_result, empty_result]
        mock_store.get_lesson_link.return_value = None

        tool = make_tool(mock_store)
        tool.execute(query="first call")
        assert len(tool.last_sources) == 1

        # Second call returns no results — last_sources should now be empty
        tool.execute(query="second call")
        assert tool.last_sources == []


# ===========================================================================
# Tool definition structure
# ===========================================================================

class TestToolDefinition:
    def test_tool_name_is_search_course_content(self, mock_store):
        tool = make_tool(mock_store)
        defn = tool.get_tool_definition()
        assert defn["name"] == "search_course_content"

    def test_tool_definition_has_description(self, mock_store):
        tool = make_tool(mock_store)
        defn = tool.get_tool_definition()
        assert "description" in defn
        assert len(defn["description"]) > 0

    def test_tool_definition_has_input_schema(self, mock_store):
        tool = make_tool(mock_store)
        defn = tool.get_tool_definition()
        assert "input_schema" in defn

    def test_query_is_required_parameter(self, mock_store):
        tool = make_tool(mock_store)
        defn = tool.get_tool_definition()
        assert "query" in defn["input_schema"]["required"]

    def test_course_name_is_optional(self, mock_store):
        tool = make_tool(mock_store)
        defn = tool.get_tool_definition()
        required = defn["input_schema"].get("required", [])
        assert "course_name" not in required

    def test_lesson_number_is_optional(self, mock_store):
        tool = make_tool(mock_store)
        defn = tool.get_tool_definition()
        required = defn["input_schema"].get("required", [])
        assert "lesson_number" not in required


# ===========================================================================
# ToolManager integration
# ===========================================================================

class TestToolManager:
    def test_register_and_execute_tool(self, mock_store, single_result):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = None

        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        result = manager.execute_tool("search_course_content", query="Python")

        assert "Python Basics" in result

    def test_execute_unknown_tool_returns_error(self, mock_store):
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="x")
        assert "not found" in result.lower() or "nonexistent_tool" in result

    def test_get_tool_definitions_returns_list(self, mock_store):
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_store))
        defs = manager.get_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) == 1

    def test_get_last_sources_returns_sources_from_tool(
        self, mock_store, single_result
    ):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = "https://example.com"

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_store)
        manager.register_tool(search_tool)
        manager.execute_tool("search_course_content", query="test")

        sources = manager.get_last_sources()
        assert len(sources) == 1

    def test_reset_sources_clears_all_tools(self, mock_store, single_result):
        mock_store.search.return_value = single_result
        mock_store.get_lesson_link.return_value = None

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_store)
        manager.register_tool(search_tool)
        manager.execute_tool("search_course_content", query="test")
        assert len(manager.get_last_sources()) == 1

        manager.reset_sources()
        assert manager.get_last_sources() == []
