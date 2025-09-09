"""Tests for AIGenerator functionality"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator"""

    def test_init(self, mock_config):
        """Test AIGenerator initialization"""
        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

        assert generator.model == mock_config.ANTHROPIC_MODEL
        assert generator.base_params["model"] == mock_config.ANTHROPIC_MODEL
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    def test_system_prompt_content(self):
        """Test that system prompt contains expected guidance"""
        prompt = AIGenerator.SYSTEM_PROMPT

        assert "search_course_content" in prompt
        assert "get_course_outline" in prompt
        assert "Up to 2 tool usage rounds" in prompt
        assert "Sequential reasoning" in prompt
        assert "tool calling" in prompt.lower() or "tools" in prompt.lower()

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_without_tools(
        self, mock_anthropic_client, mock_config, mock_anthropic_response
    ):
        """Test response generation without tool usage"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_client.return_value = mock_client

        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

        response = generator.generate_response("What is machine learning?")

        assert response == "This is a test response about MCP."
        mock_client.messages.create.assert_called_once()

        # Verify API call parameters
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["messages"][0]["content"] == "What is machine learning?"
        assert "tools" not in call_args[1]

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_tools_no_usage(
        self, mock_anthropic_client, mock_config, mock_anthropic_response, tool_manager
    ):
        """Test response generation with tools available but not used"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_client.return_value = mock_client

        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

        response = generator.generate_response(
            "What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert response == "This is a test response about MCP."

        # Verify tools were provided to API
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tool_choice"] == {"type": "auto"}

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_with_tool_usage(
        self,
        mock_anthropic_client,
        mock_config,
        mock_anthropic_tool_response,
        mock_anthropic_final_response,
        tool_manager,
    ):
        """Test response generation with tool usage"""
        mock_client = Mock()

        # First call returns tool use response, second call returns final response
        mock_client.messages.create.side_effect = [
            mock_anthropic_tool_response,
            mock_anthropic_final_response,
        ]
        mock_anthropic_client.return_value = mock_client

        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

        response = generator.generate_response(
            "What is MCP?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert (
            response
            == "Based on the course materials, MCP (Model Context Protocol) enables AI applications to securely connect to external data sources."
        )

        # Should have been called twice (initial + follow-up)
        assert mock_client.messages.create.call_count == 2

    @patch("ai_generator.anthropic.Anthropic")
    def test_handle_tool_execution(
        self,
        mock_anthropic_client,
        mock_config,
        mock_anthropic_tool_response,
        mock_anthropic_final_response,
        tool_manager,
    ):
        """Test tool execution handling"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_final_response
        mock_anthropic_client.return_value = mock_client

        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

        # Mock base parameters
        base_params = {
            "model": mock_config.ANTHROPIC_MODEL,
            "temperature": 0,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": "What is MCP?"}],
            "system": AIGenerator.SYSTEM_PROMPT,
        }

        response = generator._handle_tool_execution(
            mock_anthropic_tool_response, base_params, tool_manager
        )

        assert (
            response
            == "Based on the course materials, MCP (Model Context Protocol) enables AI applications to securely connect to external data sources."
        )

        # Verify tool was executed
        call_args = mock_client.messages.create.call_args
        messages = call_args[1]["messages"]

        # Should have 3 messages: user, assistant (tool use), user (tool results)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_sequential_tool_usage(
        self,
        mock_anthropic_client,
        mock_config,
        mock_anthropic_tool_response,
        tool_manager,
    ):
        """Test response generation with sequential tool usage (2 rounds)"""
        mock_client = Mock()

        # Mock second round: different tool use
        second_response = Mock()
        second_response.stop_reason = "tool_use"
        second_response.content = [Mock()]
        second_response.content[0].type = "tool_use"
        second_response.content[0].name = "get_course_outline"
        second_response.content[0].input = {"course_title": "Machine Learning"}
        second_response.content[0].id = "tool_2"

        # Mock final response (without tools available)
        final_response = Mock()
        final_response.stop_reason = "stop"
        final_response.content = [Mock()]
        final_response.content[0].text = (
            "Based on both searches, here's the comprehensive answer..."
        )

        mock_client.messages.create.side_effect = [
            mock_anthropic_tool_response,
            second_response,
            final_response,
        ]
        mock_anthropic_client.return_value = mock_client

        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

        response = generator.generate_response(
            "Search for a course that discusses the same topic as lesson 4 of course X",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert response == "Based on both searches, here's the comprehensive answer..."
        assert mock_client.messages.create.call_count == 3  # 2 tool rounds + 1 final

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_max_rounds_enforcement(
        self,
        mock_anthropic_client,
        mock_config,
        mock_anthropic_tool_response,
        mock_anthropic_final_response,
        tool_manager,
    ):
        """Test that tool usage is limited to 2 rounds maximum"""
        mock_client = Mock()

        # First call: tool use, Second call: tool use, Third call: final (no tools available)
        mock_client.messages.create.side_effect = [
            mock_anthropic_tool_response,
            mock_anthropic_tool_response,
            mock_anthropic_final_response,
        ]
        mock_anthropic_client.return_value = mock_client

        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

        response = generator.generate_response(
            "Complex query requiring multiple searches",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Should stop after 2 rounds even if Claude wants more tools
        expected = "Based on the course materials, MCP (Model Context Protocol) enables AI applications to securely connect to external data sources."
        assert response == expected
        assert mock_client.messages.create.call_count == 3  # 2 tool rounds + 1 final

        # Verify third call had no tools available
        final_call_args = mock_client.messages.create.call_args_list[2]
        assert "tools" not in final_call_args[1]

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_tool_then_direct_response(
        self, mock_anthropic_client, mock_config, tool_manager
    ):
        """Test tool use followed by direct response (early termination)"""
        mock_client = Mock()

        # Mock first round: tool use
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [Mock()]
        first_response.content[0].type = "tool_use"
        first_response.content[0].name = "search_course_content"
        first_response.content[0].input = {"query": "machine learning"}
        first_response.content[0].id = "tool_1"

        # Mock second round: direct response (no tool use)
        second_response = Mock()
        second_response.stop_reason = "stop"
        second_response.content = [Mock()]
        second_response.content[0].text = (
            "Based on the search results, here's the answer."
        )

        mock_client.messages.create.side_effect = [first_response, second_response]
        mock_anthropic_client.return_value = mock_client

        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

        response = generator.generate_response(
            "What is machine learning?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert response == "Based on the search results, here's the answer."
        assert (
            mock_client.messages.create.call_count == 2
        )  # 1 tool round + 1 direct response

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_context_preservation(
        self, mock_anthropic_client, mock_config, tool_manager
    ):
        """Test that messages accumulate correctly across rounds"""
        mock_client = Mock()

        # Mock first round: tool use
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [Mock()]
        first_response.content[0].type = "tool_use"
        first_response.content[0].name = "search_course_content"
        first_response.content[0].input = {"query": "lesson 1"}
        first_response.content[0].id = "tool_1"

        # Mock final response
        final_response = Mock()
        final_response.stop_reason = "stop"
        final_response.content = [Mock()]
        final_response.content[0].text = "Final answer"

        mock_client.messages.create.side_effect = [first_response, final_response]
        mock_anthropic_client.return_value = mock_client

        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

        generator.generate_response(
            "What is in lesson 1?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Check that second API call has accumulated messages
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]

        # Should have: user query, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is in lesson 1?"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"

    @patch("ai_generator.anthropic.Anthropic")
    def test_generate_response_tool_execution_error_in_sequence(
        self, mock_anthropic_client
    ):
        """Test error handling during sequential tool calls"""
        mock_client = Mock()

        # Mock first round: tool use
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [Mock()]
        first_response.content[0].type = "tool_use"
        first_response.content[0].name = "search_course_content"
        first_response.content[0].input = {"query": "test"}
        first_response.content[0].id = "tool_1"

        # Mock final response
        final_response = Mock()
        final_response.stop_reason = "stop"
        final_response.content = [Mock()]
        final_response.content[0].text = "Response despite tool error"

        mock_client.messages.create.side_effect = [first_response, final_response]
        mock_anthropic_client.return_value = mock_client

        # Create a mock tool manager that raises an error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        generator = AIGenerator("test-key", "test-model")

        response = generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content", "description": "test"}],
            tool_manager=mock_tool_manager,
        )

        # Should handle error gracefully and continue
        assert response == "Response despite tool error"
        assert mock_client.messages.create.call_count == 2

        # Verify error message was passed to Claude
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        tool_result_content = messages[2]["content"][0]["content"]
        assert "Tool execution failed" in tool_result_content

    @patch("ai_generator.anthropic.Anthropic")
    def test_conversation_history_included(
        self, mock_anthropic_client, mock_config, mock_anthropic_response
    ):
        """Test that conversation history is included in system message"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic_client.return_value = mock_client

        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

        history = "Previous conversation:\nUser: Hello\nAssistant: Hi there!"

        generator.generate_response("What is MCP?", conversation_history=history)

        # Verify history was included in system content
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert history in system_content

    @patch("ai_generator.anthropic.Anthropic")
    def test_api_error_handling(self, mock_anthropic_client, mock_config):
        """Test handling of API errors"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error: Invalid model")
        mock_anthropic_client.return_value = mock_client

        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

        with pytest.raises(Exception) as exc_info:
            generator.generate_response("What is MCP?")

        assert "API Error: Invalid model" in str(exc_info.value)

    @patch("ai_generator.anthropic.Anthropic")
    def test_tool_execution_error_handling(
        self, mock_anthropic_client, mock_config, mock_anthropic_tool_response
    ):
        """Test handling of tool execution errors"""
        mock_client = Mock()
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Response after tool error"
        mock_client.messages.create.return_value = final_response
        mock_anthropic_client.return_value = mock_client

        # Create a mock tool manager that raises an error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )

        base_params = {
            "model": mock_config.ANTHROPIC_MODEL,
            "temperature": 0,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": "What is MCP?"}],
            "system": AIGenerator.SYSTEM_PROMPT,
        }

        # Should not raise exception, should handle gracefully
        generator._handle_tool_execution(
            mock_anthropic_tool_response, base_params, mock_tool_manager
        )

        # Verify final API call was made despite tool error
        mock_client.messages.create.assert_called()


class TestAIGeneratorWithInvalidConfig:
    """Test AIGenerator with invalid configuration"""

    @patch("ai_generator.anthropic.Anthropic")
    def test_invalid_api_key(self, mock_anthropic_client):
        """Test behavior with invalid API key"""
        mock_client = Mock()
        # Simulate API authentication error
        mock_client.messages.create.side_effect = Exception(
            "Authentication failed: Invalid API key"
        )
        mock_anthropic_client.return_value = mock_client

        generator = AIGenerator("invalid-key", "claude-3-5-sonnet-20241022")

        with pytest.raises(Exception) as exc_info:
            generator.generate_response("Test query")

        assert "Authentication failed" in str(exc_info.value)

    @patch("ai_generator.anthropic.Anthropic")
    def test_invalid_model_name(self, mock_anthropic_client, mock_config):
        """Test behavior with invalid model name"""
        mock_client = Mock()
        # Simulate invalid model error
        mock_client.messages.create.side_effect = Exception(
            "Invalid model: 1claude-sonnet-4-20250514"
        )
        mock_anthropic_client.return_value = mock_client

        generator = AIGenerator(
            mock_config.ANTHROPIC_API_KEY, "1claude-sonnet-4-20250514"
        )

        with pytest.raises(Exception) as exc_info:
            generator.generate_response("Test query")

        assert "Invalid model" in str(exc_info.value)


class TestAIGeneratorIntegration:
    """Integration tests for AIGenerator"""

    def test_tool_definitions_structure(self, tool_manager):
        """Test that tool definitions are properly structured for Anthropic API"""
        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 2  # search_course_content and get_course_outline

        for definition in definitions:
            assert "name" in definition
            assert "description" in definition
            assert "input_schema" in definition
            assert definition["input_schema"]["type"] == "object"
            assert "properties" in definition["input_schema"]
            assert "required" in definition["input_schema"]

    def test_tool_manager_execution(self, tool_manager):
        """Test that tool manager can execute registered tools"""
        # Test course search tool
        result = tool_manager.execute_tool(
            "search_course_content", query="introduction"
        )
        assert len(result) > 0
        assert "MCP" in result or "introduction" in result.lower()

        # Test course outline tool
        result = tool_manager.execute_tool("get_course_outline", course_title="MCP")
        assert len(result) > 0
        assert "Course:" in result
        assert "Lessons:" in result

    def test_invalid_tool_execution(self, tool_manager):
        """Test handling of invalid tool execution"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")
        assert "Tool 'nonexistent_tool' not found" in result
