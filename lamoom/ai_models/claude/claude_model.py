from lamoom.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
import logging

from lamoom.ai_models.constants import C_4K
from lamoom.responses import AIResponse
from lamoom.ai_models.tools import ToolDefinition, AVAILABLE_TOOLS_REGISTRY, inject_tool_prompts
from enum import Enum

import typing as t
from dataclasses import dataclass

from lamoom.ai_models.utils import get_common_args

from lamoom.exceptions import RetryableCustomError, ConnectionLostError
import anthropic
from lamoom.ai_models.tools.base_tool_handler import BaseToolHandler

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
    tool_handler: BaseToolHandler = None

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

        # Initialize tool handler
        self.tool_handler = BaseToolHandler(tool_registry=AVAILABLE_TOOLS_REGISTRY)

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

    def _process_stream_response(self, 
                               client: anthropic.Anthropic,
                               current_messages_history: t.List[dict],
                               max_tokens: int,
                               system_prompt: t.Optional[str],
                               stream_function: t.Callable,
                               check_connection: t.Callable,
                               stream_params: dict,
                               mcp_call_registry: t.Dict[str, t.Callable]) -> t.Tuple[str, str, bool]:
        """Process a single streaming response from Claude.
        
        Args:
            client: Anthropic client instance
            current_messages_history: Current conversation history
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            stream_function: Function to call for each text chunk
            check_connection: Function to check connection status
            stream_params: Parameters for stream function
            mcp_call_registry: Registry of available MCP functions
            
        Returns:
            Tuple of (content, stop_reason, has_tool_call)
        """
        current_stream_part_content = ""
        stream_stop_reason = None
        has_tool_call = False
        
        call_kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": current_messages_history,
        }
        
        if system_prompt:
            call_kwargs["system"] = system_prompt

        stream_idx = 0
        try:
            logger.debug(f"Initiating Claude stream. History length: {len(current_messages_history)}")
            with client.messages.stream(**call_kwargs) as stream_handler:
                for text_chunk in stream_handler.text_stream:
                    stream_idx += 1
                    if stream_idx % 5 == 0 and not check_connection(**stream_params):
                        raise ConnectionLostError("Connection was lost!")
                    
                    current_stream_part_content += text_chunk
                    if text_chunk:
                        stream_function(text_chunk, **stream_params)
                        
                    # Check for tool call after each chunk
                    detected_tool_call = self.tool_handler.handle_tool_call(
                        current_stream_part_content
                    )
                    if detected_tool_call:
                        break  # Stop streaming when tool call is detected

                if not detected_tool_call:
                    final_message_status = stream_handler.get_final_message()
                    stream_stop_reason = final_message_status.stop_reason
                    logger.debug(f"Claude stream part finished. Stop Reason: {stream_stop_reason}")

                else:
                    current_messages_history.append(
                        {"role": "assistant", "message": current_stream_part_content + detected_tool_call.execution_result}
                    )
                    
        except ConnectionLostError:
            raise
        except anthropic.APIError as e:
            logger.exception("[CLAUDEAI] API Error during stream processing", exc_info=e)
            raise RetryableCustomError(f"Claude AI API Error: {e}") from e
        except Exception as e:
            logger.exception("Exception during Claude stream processing", exc_info=e)
            raise RetryableCustomError(f"Claude AI stream processing failed: {e}") from e

        return current_stream_part_content, stream_stop_reason, has_tool_call

    def _process_non_stream_response(self,
                                   client: anthropic.Anthropic,
                                   current_messages_history: t.List[dict],
                                   max_tokens: int,
                                   system_prompt: t.Optional[str]) -> str:
        """Process a non-streaming response from Claude.
        
        Args:
            client: Anthropic client instance
            current_messages_history: Current conversation history
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            Final response content
        """
        call_kwargs = {
            "model": self.model,
            "messages": current_messages_history,
            "max_tokens": max_tokens,
        }
        
        if system_prompt:
            call_kwargs['system'] = system_prompt

        logger.debug(f"Calling Claude API with messages: {current_messages_history}")
        response = client.messages.create(**call_kwargs)
        
        text_blocks = [block.text for block in response.content if block.type == "text"]
        content = "\n".join(text_blocks).strip()
        
        # Check for tool call
        content, has_tool_call = self.tool_handler.handle_tool_call(content)
        
        if has_tool_call:
            # Recursively process the next response
            return self._process_non_stream_response(client, current_messages_history, max_tokens, system_prompt)
            
        return content

    def call(self, 
            messages: t.List[dict], 
            max_tokens: int, 
            client_secrets: dict = {}, 
            tool_registry: t.Dict[str, ToolDefinition] = {},
            mcp_call_registry: t.Dict[str, t.Callable] = None,
            max_tool_iterations: int = 5,   # Safety limit for sequential calls
            **kwargs) -> AIResponse:
        max_tokens = min(max_tokens, self.max_tokens)
        
        common_args = get_common_args(max_tokens)
        kwargs = {
            **common_args,
            **self.get_params(),
            **kwargs,
        }
        
        # Update tool handler with MCP registry
        if mcp_call_registry:
            self.tool_handler.mcp_call_registry = mcp_call_registry
            
        # Inject Tool Prompts into initial messages
        current_messages_history = inject_tool_prompts(messages, list(tool_registry.values()))
        
        system_prompt = None
        if current_messages_history and current_messages_history[0].get('role') == "system":
            system_prompt = current_messages_history[0].get('content')
            current_messages_history = current_messages_history[1:]
            
        logger.debug(
            f"Calling {current_messages_history} with max_tokens {max_tokens} and kwargs {kwargs}"
        )
        client = self.get_client(client_secrets)

        stream_function = kwargs.get("stream_function")
        check_connection = kwargs.get("check_connection")
        stream_params = kwargs.get("stream_params")

        content = ""

        try:
            if kwargs.get("stream"):
                iteration_count = 0
                while iteration_count < max_tool_iterations:
                    iteration_count += 1
                    logger.info(f"--- Custom Claude Tool Streaming Iteration: {iteration_count} ---")

                    current_stream_part_content, stream_stop_reason, has_tool_call = self._process_stream_response(
                        client, current_messages_history, max_tokens, system_prompt,
                        stream_function, check_connection, stream_params, mcp_call_registry
                    )

                    # Add the raw assistant response to history *before* parsing
                    assistant_message_to_add = {"role": "assistant", "content": current_stream_part_content}
                    current_messages_history.append(assistant_message_to_add)

                    if has_tool_call:
                        continue

                    # --- No Tool Call Found in this stream part ---
                    content += current_stream_part_content
                    if stream_stop_reason == "end_turn":
                        break

            else:
                content = self._process_non_stream_response(
                    client, current_messages_history, max_tokens, system_prompt
                )

        except Exception as e:
            logger.exception("Exception during Claude API call", exc_info=e)
            raise RetryableCustomError(f"Claude AI API call failed: {e}") from e

        return AIResponse(content=content)

    @property
    def name(self) -> str:
        return f"Claude {self.family}"

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
        }

    def get_metrics_data(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
            "family": self.family,
            "max_tokens": self.max_tokens,
        }
