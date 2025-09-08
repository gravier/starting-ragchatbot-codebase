from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **search_course_content**: Search within course materials for specific content
2. **get_course_outline**: Retrieve complete course outline with all lessons

Tool Usage Guidelines:
- **Content questions**: Use search_course_content for questions about specific topics, lessons, or detailed materials
- **Outline questions**: Use get_course_outline for questions about course structure, lesson lists, or complete overviews
- **Up to 2 tool usage rounds maximum** - You can use tools in up to 2 separate rounds to gather comprehensive information
- **Sequential reasoning**: Use first tool call to gather initial information, then optionally use second call to gather additional context
- Synthesize all tool results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search_course_content tool first, then answer
- **Course outline questions**: Use get_course_outline tool first, then answer
- **Complex queries**: May require multiple tool calls - use first call to gather basic info, second call for additional context
- **No meta-commentary**: Provide direct answers only — no reasoning process, search explanations, or question-type analysis

For Course Outlines:
- Always include the course title and course link when available
- List all lessons with their numbers and titles in order
- Format clearly for easy reading

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
        Generate AI response with optional sequential tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Initialize conversation with user query
        messages = [{"role": "user", "content": query}]
        max_rounds = 2

        # Build base system content
        base_system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Sequential tool calling loop
        round_num = 0
        while round_num < max_rounds:
            # Update system prompt with round information
            system_content = f"{base_system_content}\n\nCurrent tool usage round: {round_num + 1} of {max_rounds}"

            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages.copy(),  # Use copy to avoid mutation
                "system": system_content,
            }

            # Add tools if we haven't reached max rounds and have tools/tool_manager
            tools_available = tools and tool_manager and round_num < max_rounds
            if tools_available:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            try:
                # Make API call
                response = self.client.messages.create(**api_params)

                # Check if Claude wants to use tools and we haven't exceeded max rounds
                if response.stop_reason == "tool_use" and tools_available:
                    # Execute tools and continue to next round
                    messages.append({"role": "assistant", "content": response.content})

                    tool_results = self._execute_tools_for_round(response, tool_manager)
                    messages.append({"role": "user", "content": tool_results})

                    round_num += 1
                    continue  # Go to next round
                else:
                    # Claude provided final response or no more tools available
                    return response.content[0].text

            except Exception as e:
                # Handle API errors gracefully
                if round_num == 0:
                    raise e  # Re-raise on first round
                else:
                    # Return best response we have so far
                    return f"An error occurred during tool usage: {str(e)}"

        # We've used max rounds, make final call without tools
        final_system_content = (
            f"{base_system_content}\n\nFinal response - no more tool usage allowed"
        )
        final_api_params = {
            **self.base_params,
            "messages": messages.copy(),
            "system": final_system_content,
        }

        try:
            final_response = self.client.messages.create(**final_api_params)
            return final_response.content[0].text
        except Exception as e:
            return f"An error occurred during final response: {str(e)}"

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        DEPRECATED: This method is maintained for backward compatibility.
        New sequential tool calling logic is handled in generate_response().

        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute tools using the new helper method
        tool_results = self._execute_tools_for_round(initial_response, tool_manager)
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _execute_tools_for_round(self, response, tool_manager):
        """
        Execute all tool calls for a single round and return results.

        Args:
            response: The response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            List of tool results for the conversation
        """
        tool_results = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )
                except Exception as e:
                    tool_result = f"Tool execution failed: {str(e)}"

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        return tool_results
