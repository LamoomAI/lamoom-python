from lamoom.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
import logging

from lamoom.ai_models.constants import C_200K, C_4K
from lamoom.responses import AIResponse
from lamoom.ai_models.tools import ToolDefinition, AVAILABLE_TOOLS
from decimal import Decimal
from enum import Enum
import json

import typing as t
from dataclasses import dataclass, is_dataclass, asdict

from lamoom.ai_models.claude.responses import ClaudeAIReponse
from lamoom.ai_models.claude.constants import HAIKU, SONNET, OPUS
from lamoom.ai_models.utils import get_common_args

from openai.types.chat import ChatCompletionMessage as Message
from lamoom.responses import Prompt
from lamoom.exceptions import RetryableCustomError, ConnectionLostError
import anthropic
from anthropic.types import ToolParam, ToolUseBlock, ToolResultBlockParam, ContentBlock, MessageParam

logger = logging.getLogger(__name__)


class FamilyModel(Enum):
    haiku = "Claude 3 Haiku"
    sonnet = "Claude 3 Sonnet"
    opus = "Claude 3 Opus"
    

@dataclass(kw_only=True)
class ClaudeAIModel(AIModel):
    model: str
    max_tokens: int = C_4K
    api_key: str = None
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.CLAUDE
    family: str = None

    def __post_init__(self):
        if HAIKU in self.model:
            self.family = FamilyModel.haiku.value
        elif SONNET in self.model:
            self.family = FamilyModel.sonnet.value
        elif OPUS in self.model:
            self.family = FamilyModel.opus.value
        else:
            logger.info(
                f"Unknown family for {self.model}. Please add it obviously. Setting as Claude 3 Opus"
            )
            self.family = FamilyModel.opus.value

        logger.debug(f"Initialized ClaudeAIModel: {self}")

    def get_client(self, client_secrets: dict) -> anthropic.Anthropic:
        return anthropic.Anthropic(api_key=client_secrets.get("api_key"))

    def uny_all_messages_with_same_role(self, messages: t.List[dict]) -> t.List[dict]:
        result = []
        last_role = None
        for message in messages:
            if message.get("role") == "system":
                message["role"] = "user"
            if last_role != message.get("role"):
                result.append(message)
                last_role = message.get("role")
            else:
                result[-1]["content"] += message.get("content")
        return result
    
    def _format_tools_for_anthropic(self, tools: t.List[ToolDefinition]) -> t.List[t.Dict[str, t.Any]]:
        """Converts generic ToolDefinition list to Anthropic's tool format."""
        anthropic_tools = []
        for tool_def in tools:
            properties = {}
            required_params = []
            for param in tool_def.parameters:
                param_type = "string" # Default
                if param.type == "number": param_type = "number"
                elif param.type == "boolean": param_type = "boolean"
                elif param.type == "integer": param_type = "integer"
                # Add other types ('array', 'object') if needed
                properties[param.name] = {"type": param_type, "description": param.description}
                if param.required: required_params.append(param.name)

            anthropic_tools.append({
                "name": tool_def.name,
                "description": tool_def.description,
                "input_schema": {"type": "object", "properties": properties, "required": required_params},
            })
        return anthropic_tools

    def _execute_anthropic_tools(
        self,
        tool_use_blocks: t.List[ToolUseBlock], # Use the specific Anthropic type
        tool_registry: t.Dict[str, ToolDefinition]
        ) -> t.List[MessageParam]:
        """Executes tools based on Anthropic's tool_use blocks and returns tool_result messages."""
        tools_result_message = {"role": "user", "content": []}
        tools_content = []
        
        for tool_use in tool_use_blocks:
            # Ensure it's actually a tool_use block (safety check)
            if not isinstance(tool_use, ToolUseBlock):
                logger.warning(f"Skipping non-ToolUseBlock item in tool execution: {type(tool_use)}")
                continue

            tool_name = tool_use.name
            tool_use_id = tool_use.id
            tool_input = tool_use.input or {} # Input is already a dict
            tool_result_content_str = "" # Result must be a string

            logger.info(f"Attempting tool execution for '{tool_name}' (ID: {tool_use_id}) with input: {tool_input}")

            tool_definition = tool_registry.get(tool_name)
            if tool_definition and tool_definition.execution_function:
                try:
                    tool_result_content_str = tool_definition.execution_function(**tool_input)
                    logger.info(f"Tool '{tool_name}' executed successfully. Result snippet: {tool_result_content_str[:200]}...")
                    result_block = {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": tool_result_content_str
                    }

                except Exception as tool_exc:
                    logger.exception(f"Error executing tool '{tool_name}'", exc_info=tool_exc)
                    error_content = f"Execution of tool '{tool_name}' failed: {str(tool_exc)}"
                    result_block = {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": error_content
                    }
            else:
                logger.warning(f"Tool '{tool_name}' requested by model but not found in registry or not executable.")
                error_content = f"Tool '{tool_name}' is not available or implemented."
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": error_content
                }

            # Append the result as a user message containing the tool_result block
            tools_content.append(result_block)

        tools_result_message['content'] = tools_content
        return tools_result_message


    def call(self, 
             messages: t.List[dict], 
             max_tokens: int, 
             client_secrets: dict = {}, 
             available_tools: t.List[ToolDefinition] = AVAILABLE_TOOLS,
             **kwargs) -> AIResponse:
        max_tokens = min(max_tokens, self.max_tokens)
        
        common_args = get_common_args(max_tokens)
        kwargs = {
            **common_args,
            **self.get_params(),
            **kwargs,
        }
        messages = self.uny_all_messages_with_same_role(messages)

        logger.debug(
            f"Calling {messages} with max_tokens {max_tokens} and kwargs {kwargs}"
        )
        client = self.get_client(client_secrets)

        stream_function = kwargs.get("stream_function")
        check_connection = kwargs.get("check_connection")
        stream_params = kwargs.get("stream_params")

        content = ""
        
        tool_registry = {} # Dictionary to hold tool definitions for easy access
        # Populate the tool registry with available tools
        for tool in available_tools:
            tool_registry[tool.name] = tool
        
        anthropic_formatted_tools = []
        if available_tools:
            anthropic_formatted_tools = self._format_tools_for_anthropic(available_tools)
            logger.info(f"Formatted tools for Anthropic: {anthropic_formatted_tools}")

        try:
            if kwargs.get("stream"):
                with client.messages.stream(
                    model=self.model, max_tokens=max_tokens, messages=messages
                ) as stream:
                    idx = 0
                    for text in stream.text_stream:
                        if idx % 5 == 0:
                            if not check_connection(**stream_params):
                                raise ConnectionLostError("Connection was lost!")

                        stream_function(text, **stream_params)
                        content += text
                        idx += 1
            else:
                initial_user_messages = messages # Keep track of original request messages
                
                current_messages_history = initial_user_messages.copy()
                iteration_count = 0
                max_iterations = 5 # Safety break for sequential tool calls
                final_response_content = ""
                final_stop_reason = None
            
                # Can be while True:
                while iteration_count < max_iterations:
                    
                    call_kwargs = {
                        "model": self.model,
                        "messages": current_messages_history,
                        "max_tokens": max_tokens,
                    }
    
                    logger.debug(f"Calling Claude API with messages: {current_messages_history}")
                    
                    if anthropic_formatted_tools:
                        call_kwargs["tools"] = anthropic_formatted_tools
                        
                    response = client.messages.create(**call_kwargs)
                    final_stop_reason = response.stop_reason
                    logger.debug(f"Non-streaming response received. Stop Reason: {final_stop_reason}, Role: {response.role}")
                    
                    if final_stop_reason == "tool_use":
                        logger.info(f"Tool use requested (non-streaming).")
                        current_messages_history.append({'role': response.role, 'content': response.content})
                        tool_use_blocks: t.List[ToolUseBlock] = [block for block in response.content if isinstance(block, ToolUseBlock)]
                        
                        logger.debug(f"Tool blocks: {tool_use_blocks}")
                        
                        if not tool_use_blocks:
                            logger.warning("Stop reason was 'tool_use', but no ToolUseBlocks found in content.")
                            break
                        
                        tools_result_message = self._execute_anthropic_tools(tool_use_blocks, tool_registry)
                        current_messages_history.append(tools_result_message)
                        continue
                    else:
                        logger.info(f"Model finished without requesting tools (Reason: {final_stop_reason}).")
                        text_blocks = [block.text for block in response.content if block.type == "text"]
                        final_response_content = "\n".join(text_blocks).strip()
                        break # Exit the while loop, interaction is complete
                    
            content = final_response_content
            
            return ClaudeAIReponse(
                message=Message(content=content, role="assistant"),
                content=content,
                prompt=Prompt(
                    messages=kwargs.get("messages"),
                    functions=kwargs.get("tools"),
                    max_tokens=max_tokens,
                    temperature=kwargs.get("temperature"),
                    top_p=kwargs.get("top_p"),
                ),
            )
        except Exception as e:
            logger.exception("[CLAUDEAI] failed to handle chat stream", exc_info=e)
            raise RetryableCustomError(f"Claude AI call failed!")

    @property
    def name(self) -> str:
        return self.model

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
        }

    def get_metrics_data(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
        }
