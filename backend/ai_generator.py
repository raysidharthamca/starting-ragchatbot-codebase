import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Available Tools:
- **search_course_content**: Use for questions about specific course content, concepts, or detailed lesson material
- **get_course_outline**: Use for outline or structure questions (e.g. "what lessons are in X?", "list the topics in X", "give me an overview of X course"). Return the course title, course link, and a numbered list of all lessons with their titles.

Tool Usage Rules:
- **Multi-step tool use**: You may make up to 2 sequential tool calls per query when the answer requires gathering information from multiple searches (e.g. retrieving a course outline first, then searching for related content). Each tool call will be executed and its results returned before you proceed.
- After all tool calls are complete, synthesize the results into a single, accurate response
- Synthesize results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager, tools)

        # Return direct response
        return response.content[0].text

    def _handle_tool_execution(
        self, current_response, base_params: Dict[str, Any], tool_manager, tools
    ):
        """
        Handle sequential tool execution with up to MAX_TOOL_ROUNDS rounds.

        Each round appends the assistant's tool-use response and tool results to the
        growing messages list, then makes another API call with tools if rounds remain.
        If an intermediate call returns text (no tool_use), exits early without a
        separate synthesis call. After all rounds, a final synthesis call without tools
        is made.

        Args:
            current_response: The initial tool_use response from generate_response
            base_params: API parameters from the original call (includes messages, system)
            tool_manager: Manager to execute tools
            tools: Tool definitions to re-attach on intermediate round calls

        Returns:
            Final response text
        """
        messages = base_params["messages"].copy()
        round_count = 0

        while round_count < self.MAX_TOOL_ROUNDS:
            # Append assistant's tool-use content to conversation
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool_use blocks; catch errors and forward as result strings
            tool_results = []
            for block in current_response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                    except Exception as e:
                        result = str(e)
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": result}
                    )

            # Safety: no tool_use blocks found despite tool_use stop_reason
            if not tool_results:
                break

            messages.append({"role": "user", "content": tool_results})
            round_count += 1

            if round_count < self.MAX_TOOL_ROUNDS:
                # Make intermediate call WITH tools — Claude may request another round
                mid_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": base_params["system"],
                    "tools": tools,
                    "tool_choice": {"type": "auto"},
                }
                mid_response = self.client.messages.create(**mid_params)
                if mid_response.stop_reason != "tool_use":
                    # Early exit: Claude synthesized naturally, no extra synthesis call needed
                    return mid_response.content[0].text
                current_response = mid_response
            # If round_count == MAX_TOOL_ROUNDS, loop exits — synthesis call follows

        # Final synthesis call without tools
        final_params = {**self.base_params, "messages": messages, "system": base_params["system"]}
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
