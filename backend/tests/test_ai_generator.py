"""
Tests for AIGenerator — verifying:
1. Direct (non-tool) responses are returned correctly.
2. Tool-use responses trigger ToolManager.execute_tool with correct arguments.
3. Tool results are fed back to the Claude API as a follow-up message.
4. The final text from the follow-up call is what gets returned.
5. Conversation history is embedded in the system prompt.
6. The configured model name is valid for the Anthropic API.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from ai_generator import AIGenerator
from config import config

# ---------------------------------------------------------------------------
# Helpers — build realistic mock Anthropic SDK responses
# ---------------------------------------------------------------------------


def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(name: str, tool_id: str, input_dict: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = input_dict
    return block


def _make_response(stop_reason: str, content_blocks: list):
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content_blocks
    return response


def _make_generator(mock_client=None):
    """Return an AIGenerator whose Anthropic client is replaced with a mock."""
    gen = AIGenerator(api_key="test-key", model="claude-sonnet-4-5")
    if mock_client is not None:
        gen.client = mock_client
    return gen


# ---------------------------------------------------------------------------
# 1. Direct text response (no tool use)
# ---------------------------------------------------------------------------


class TestDirectResponse:
    def test_returns_text_from_end_turn_response(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response(
            "end_turn", [_text_block("Here is your answer.")]
        )

        gen = _make_generator(client)
        result = gen.generate_response(query="What is 2+2?")

        assert result == "Here is your answer."

    def test_no_tool_manager_call_on_direct_response(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response(
            "end_turn", [_text_block("Direct answer")]
        )
        tool_manager = MagicMock()

        gen = _make_generator(client)
        gen.generate_response(query="General question", tool_manager=tool_manager)

        tool_manager.execute_tool.assert_not_called()

    def test_api_called_exactly_once_on_direct_response(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response("end_turn", [_text_block("Answer")])

        gen = _make_generator(client)
        gen.generate_response(query="q")

        assert client.messages.create.call_count == 1

    def test_query_appears_in_messages(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response("end_turn", [_text_block("ok")])

        gen = _make_generator(client)
        gen.generate_response(query="My special question")

        call_kwargs = client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        assert any("My special question" in m.get("content", "") for m in messages)


# ---------------------------------------------------------------------------
# 2. Tool use — trigger and argument forwarding
# ---------------------------------------------------------------------------


class TestToolUse:
    def _setup(self, tool_input: dict, tool_result: str = "search results"):
        """
        Set up a scenario where the first API call triggers tool use
        and the second returns a final text response.
        """
        first_response = _make_response(
            "tool_use",
            [_tool_use_block("search_course_content", "tool_abc", tool_input)],
        )
        second_response = _make_response("end_turn", [_text_block("Final synthesized answer.")])

        client = MagicMock()
        client.messages.create.side_effect = [first_response, second_response]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = tool_result

        gen = _make_generator(client)
        return gen, client, tool_manager

    def test_execute_tool_is_called_on_tool_use(self):
        gen, client, tool_manager = self._setup({"query": "MCP basics"})
        gen.generate_response(query="What is MCP?", tools=[{}], tool_manager=tool_manager)
        tool_manager.execute_tool.assert_called_once()

    def test_execute_tool_called_with_correct_name(self):
        gen, client, tool_manager = self._setup({"query": "MCP basics"})
        gen.generate_response(query="What is MCP?", tools=[{}], tool_manager=tool_manager)
        name_arg = tool_manager.execute_tool.call_args[0][0]
        assert name_arg == "search_course_content"

    def test_execute_tool_called_with_correct_kwargs(self):
        gen, client, tool_manager = self._setup(
            {"query": "MCP basics", "course_name": "MCP Course"}
        )
        gen.generate_response(query="What is MCP?", tools=[{}], tool_manager=tool_manager)
        _, kwargs = tool_manager.execute_tool.call_args
        assert kwargs.get("query") == "MCP basics"
        assert kwargs.get("course_name") == "MCP Course"

    def test_final_text_is_returned_after_tool_use(self):
        gen, client, tool_manager = self._setup({"query": "MCP"})
        result = gen.generate_response(query="What is MCP?", tools=[{}], tool_manager=tool_manager)
        assert result == "Final synthesized answer."

    def test_api_called_twice_on_tool_use(self):
        gen, client, tool_manager = self._setup({"query": "topic"})
        gen.generate_response(query="q", tools=[{}], tool_manager=tool_manager)
        assert client.messages.create.call_count == 2

    def test_tool_result_present_in_follow_up_messages(self):
        gen, client, tool_manager = self._setup({"query": "topic"}, "SEARCH RESULT TEXT")
        gen.generate_response(query="q", tools=[{}], tool_manager=tool_manager)

        # Second API call should have the tool result in messages
        second_call_kwargs = client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Flatten all message content to a single string for easy assertion
        content_str = str(messages)
        assert "SEARCH RESULT TEXT" in content_str

    def test_tool_use_id_forwarded_in_tool_result(self):
        gen, client, tool_manager = self._setup({"query": "topic"})
        gen.generate_response(query="q", tools=[{}], tool_manager=tool_manager)

        second_call_kwargs = client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        content_str = str(messages)
        assert "tool_abc" in content_str

    def test_follow_up_call_includes_tools_for_potential_round_two(self):
        """The intermediate API call after round 1 must include tools so Claude
        can optionally make a second tool call. If Claude returns text, the loop
        exits early without a separate synthesis call (early-exit path)."""
        gen, client, tool_manager = self._setup({"query": "topic"})
        gen.generate_response(
            query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        second_call_kwargs = client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs
        assert second_call_kwargs.get("tool_choice") == {"type": "auto"}


# ---------------------------------------------------------------------------
# 3. Conversation history in system prompt
# ---------------------------------------------------------------------------


class TestConversationHistory:
    def test_history_included_in_system_when_provided(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response("end_turn", [_text_block("ok")])

        gen = _make_generator(client)
        gen.generate_response(
            query="follow-up question",
            conversation_history="User: hello\nAssistant: hi",
        )

        call_kwargs = client.messages.create.call_args[1]
        system = call_kwargs["system"]
        assert "User: hello" in system
        assert "Assistant: hi" in system

    def test_no_history_key_absent_from_system(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response("end_turn", [_text_block("ok")])

        gen = _make_generator(client)
        gen.generate_response(query="question", conversation_history=None)

        call_kwargs = client.messages.create.call_args[1]
        system = call_kwargs["system"]
        # System prompt should still exist but not contain "Previous conversation"
        # when there is no history
        assert "Previous conversation:" not in system or "None" not in system


# ---------------------------------------------------------------------------
# 4. Tools wiring
# ---------------------------------------------------------------------------


class TestToolsWiring:
    def test_tools_added_to_api_params_when_provided(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response("end_turn", [_text_block("ok")])
        tool_def = {"name": "search_course_content", "input_schema": {}}

        gen = _make_generator(client)
        gen.generate_response(query="q", tools=[tool_def])

        call_kwargs = client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == [tool_def]

    def test_tool_choice_auto_when_tools_provided(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response("end_turn", [_text_block("ok")])

        gen = _make_generator(client)
        gen.generate_response(query="q", tools=[{"name": "t"}])

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs.get("tool_choice") == {"type": "auto"}

    def test_no_tool_choice_when_no_tools(self):
        client = MagicMock()
        client.messages.create.return_value = _make_response("end_turn", [_text_block("ok")])

        gen = _make_generator(client)
        gen.generate_response(query="q")

        call_kwargs = client.messages.create.call_args[1]
        assert "tool_choice" not in call_kwargs


# ---------------------------------------------------------------------------
# 5. Model name validity
# ---------------------------------------------------------------------------

KNOWN_VALID_MODELS = {
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
}


class TestModelName:
    def test_configured_model_is_a_known_valid_model(self):
        """
        config.ANTHROPIC_MODEL must be a recognized Anthropic model ID.
        An unrecognised model name causes the API to return a 404/400 error,
        which propagates as an exception → 500 from the FastAPI endpoint →
        'Query failed' in the browser.
        """
        assert config.ANTHROPIC_MODEL in KNOWN_VALID_MODELS, (
            f"Model '{config.ANTHROPIC_MODEL}' is not in the known-valid model list.\n"
            f"Valid options: {sorted(KNOWN_VALID_MODELS)}\n"
            "Update ANTHROPIC_MODEL in backend/config.py to a valid model ID."
        )

    def test_model_name_does_not_contain_date_suffix_in_wrong_format(self):
        """
        New-style model names (claude-sonnet-4-5, claude-opus-4-6) do NOT
        have a YYYYMMDD date suffix.  Old-style names (claude-3-*) do.
        Mixing them (e.g. 'claude-sonnet-4-20250514') is invalid.
        """
        model = config.ANTHROPIC_MODEL
        parts = model.split("-")
        # If model starts with 'claude-3' it may have a date suffix — that is fine.
        # New claude-4+ models should NOT end with an 8-digit date.
        if parts[1] != "3":
            last_part = parts[-1]
            assert not (last_part.isdigit() and len(last_part) == 8), (
                f"Model '{model}' looks like a new-style model but has an old-style "
                "date suffix (YYYYMMDD). Remove the date suffix or use a valid model ID."
            )


# ---------------------------------------------------------------------------
# 6. Sequential tool use (up to 2 rounds)
# ---------------------------------------------------------------------------


class TestSequentialToolUse:
    """
    Verify the multi-round tool loop: up to 2 tool-call rounds each as a
    separate API request, with full context preserved between rounds.
    """

    def _setup_two_rounds(self, result1="RESULT_1", result2="RESULT_2"):
        """
        Queue 3 responses: tool_use → tool_use → end_turn.
        Returns (gen, client, tool_manager).
        """
        r1 = _make_response(
            "tool_use", [_tool_use_block("search_course_content", "id_r1", {"query": "first"})]
        )
        r2 = _make_response(
            "tool_use", [_tool_use_block("get_course_outline", "id_r2", {"course_name": "X"})]
        )
        r3 = _make_response("end_turn", [_text_block("Final answer after two rounds.")])

        client = MagicMock()
        client.messages.create.side_effect = [r1, r2, r3]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = [result1, result2]

        gen = _make_generator(client)
        return gen, client, tool_manager

    def test_two_round_tool_use_makes_three_api_calls(self):
        gen, client, tool_manager = self._setup_two_rounds()
        gen.generate_response(query="q", tools=[{}], tool_manager=tool_manager)
        assert client.messages.create.call_count == 3

    def test_two_round_tool_use_returns_final_text(self):
        gen, client, tool_manager = self._setup_two_rounds()
        result = gen.generate_response(query="q", tools=[{}], tool_manager=tool_manager)
        assert result == "Final answer after two rounds."

    def test_two_round_tool_use_executes_tools_twice(self):
        gen, client, tool_manager = self._setup_two_rounds()
        gen.generate_response(query="q", tools=[{}], tool_manager=tool_manager)
        assert tool_manager.execute_tool.call_count == 2

    def test_round_two_api_call_includes_tools(self):
        """The intermediate (round-2) API call must include tools so Claude can make another call."""
        gen, client, tool_manager = self._setup_two_rounds()
        gen.generate_response(
            query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )
        second_call_kwargs = client.messages.create.call_args_list[1][1]
        assert "tools" in second_call_kwargs
        assert second_call_kwargs.get("tool_choice") == {"type": "auto"}

    def test_synthesis_call_omits_tools(self):
        """The final (synthesis) API call must NOT include tools."""
        gen, client, tool_manager = self._setup_two_rounds()
        gen.generate_response(
            query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )
        third_call_kwargs = client.messages.create.call_args_list[2][1]
        assert "tools" not in third_call_kwargs
        assert "tool_choice" not in third_call_kwargs

    def test_messages_accumulate_across_rounds(self):
        """Both tool results must appear in the synthesis call's messages."""
        gen, client, tool_manager = self._setup_two_rounds("RESULT_ROUND_1", "RESULT_ROUND_2")
        gen.generate_response(query="q", tools=[{}], tool_manager=tool_manager)
        third_call_kwargs = client.messages.create.call_args_list[2][1]
        content_str = str(third_call_kwargs["messages"])
        assert "RESULT_ROUND_1" in content_str
        assert "RESULT_ROUND_2" in content_str

    def test_early_termination_when_round_two_returns_text(self):
        """If the intermediate call returns end_turn, no separate synthesis call is made."""
        r1 = _make_response(
            "tool_use", [_tool_use_block("search_course_content", "id1", {"query": "q"})]
        )
        r2 = _make_response("end_turn", [_text_block("Synthesized early.")])

        client = MagicMock()
        client.messages.create.side_effect = [r1, r2]

        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "some result"

        gen = _make_generator(client)
        result = gen.generate_response(query="q", tools=[{}], tool_manager=tool_manager)

        assert client.messages.create.call_count == 2
        assert result == "Synthesized early."


# ---------------------------------------------------------------------------
# 7. Tool error handling
# ---------------------------------------------------------------------------


class TestToolErrorHandling:
    """
    Verify that tool execution errors are caught and forwarded to Claude
    as the tool result string rather than propagating as exceptions.
    """

    def _setup_with_error(self, error_msg="DB connection failed"):
        r1 = _make_response(
            "tool_use", [_tool_use_block("search_course_content", "id_err", {"query": "x"})]
        )
        r2 = _make_response("end_turn", [_text_block("Sorry, the tool failed.")])

        client = MagicMock()
        client.messages.create.side_effect = [r1, r2]

        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError(error_msg)

        gen = _make_generator(client)
        return gen, client, tool_manager

    def test_tool_error_does_not_raise_exception_to_caller(self):
        gen, client, tool_manager = self._setup_with_error()
        result = gen.generate_response(query="q", tools=[{}], tool_manager=tool_manager)
        assert isinstance(result, str)

    def test_tool_exception_passed_as_error_string_in_result(self):
        gen, client, tool_manager = self._setup_with_error("DB connection failed")
        gen.generate_response(query="q", tools=[{}], tool_manager=tool_manager)
        second_call_kwargs = client.messages.create.call_args_list[1][1]
        content_str = str(second_call_kwargs["messages"])
        assert "DB connection failed" in content_str
