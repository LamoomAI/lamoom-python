from lamoom.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
import logging

from lamoom.ai_models.constants import C_200K, C_4K
from lamoom.responses import AIResponse
from lamoom.ai_models.tools import ToolDefinition, AVAILABLE_TOOLS_REGISTRY, inject_tool_prompts, parse_tool_call_block, \
    format_tool_result_message

from enum import Enum
import json

import typing as t
from dataclasses import dataclass

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
        """Unify consecutive messages with the same role."""
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

    def call(self, 
            messages: t.List[dict], 
            max_tokens: int, 
            client_secrets: dict = {}, 
            tool_registry: t.Dict[str, ToolDefinition] = AVAILABLE_TOOLS_REGISTRY,
            max_tool_iterations: int = 10,   # Safety limit for sequential calls
            **kwargs) -> AIResponse:
        """Main entry point for calling the Claude AI model."""
        try:
            # Prepare inputs
            prepared_data = self._prepare_request_data(messages, max_tokens, kwargs)
            max_tokens = prepared_data["max_tokens"]
            current_messages_history = prepared_data["messages_history"]
            system_prompt = prepared_data["system_prompt"]
            
            # Get client
            client = self.get_client(client_secrets)
            
            # Handle streaming or regular request
            if kwargs.get("stream"):
                content = self._handle_streaming_request(
                    client=client,
                    current_messages_history=current_messages_history,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    tool_registry=tool_registry,
                    max_tool_iterations=max_tool_iterations,
                    stream_function=kwargs.get("stream_function"),
                    check_connection=kwargs.get("check_connection"),
                    stream_params=kwargs.get("stream_params")
                )
            else:
                content = self._handle_standard_request(
                    client=client,
                    current_messages_history=current_messages_history,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    tool_registry=tool_registry,
                    max_tool_iterations=max_tool_iterations
                )
                
            # Create response
            return self._create_response(content, max_tokens, kwargs)
            
        except Exception as e:
            logger.exception("[CLAUDEAI] failed to handle chat request", exc_info=e)
            raise RetryableCustomError(f"Claude AI call failed!") from e

    def _prepare_request_data(self, messages: t.List[dict], max_tokens: int, kwargs: dict) -> dict:
        """Prepare the request data - messages, system prompt, and parameters."""
        # Adjust max_tokens
        max_tokens = min(max_tokens, self.max_tokens)
        
        # Prepare common arguments
        common_args = get_common_args(max_tokens)
        kwargs = {
            **common_args,
            **self.get_params(),
            **kwargs,
        }
        
        # Process messages
        unified_messages = self.uny_all_messages_with_same_role(messages)
        tool_definitions = list(AVAILABLE_TOOLS_REGISTRY.values())
        current_messages_history = inject_tool_prompts(unified_messages, tool_definitions)
        
        # Extract system prompt if present
        system_prompt = None
        if current_messages_history and current_messages_history[0].get('role') == "system":
            system_prompt = current_messages_history[0].get('content')
            current_messages_history = current_messages_history[1:]
            
        logger.debug(f"Prepared request with messages: {current_messages_history}, max_tokens: {max_tokens}")
        
        return {
            "max_tokens": max_tokens,
            "messages_history": current_messages_history,
            "system_prompt": system_prompt,
            "kwargs": kwargs
        }

    def _handle_streaming_request(
        self, 
        client: anthropic.Anthropic,
        current_messages_history: t.List[dict],
        system_prompt: t.Optional[str],
        max_tokens: int,
        tool_registry: t.Dict[str, ToolDefinition],
        max_tool_iterations: int,
        stream_function: t.Callable,
        check_connection: t.Callable,
        stream_params: dict
    ) -> str:
        """Handle streaming requests to Claude AI."""
        iteration_count = 0
        final_content = ""
        
        while iteration_count < max_tool_iterations:
            iteration_count += 1
            logger.info(f"--- Custom Claude Tool Streaming Iteration: {iteration_count} ---")

            # Process single stream iteration
            stream_result = self._process_stream_iteration(
                client=client,
                current_messages_history=current_messages_history,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                stream_function=stream_function,
                check_connection=check_connection,
                stream_params=stream_params
            )
            
            current_stream_part_content = stream_result["content"]
            
            # Add to message history
            assistant_message = {"role": "assistant", "content": current_stream_part_content}
            current_messages_history.append(assistant_message)
            
            # Check for tool calls
            parsed_tool_call = parse_tool_call_block(current_stream_part_content)
            
            if parsed_tool_call:
                # Handle tool call
                self._handle_tool_call(
                    tool_call=parsed_tool_call,
                    tool_registry=tool_registry,
                    messages_history=current_messages_history
                )
                continue
            else:
                # No tool call - we're done
                logger.info("No custom tool call block found in this stream part. Finishing.")
                final_content = current_stream_part_content
                break
                
        # Check if we reached max iterations
        if iteration_count >= max_tool_iterations:
            logger.warning(
                f"Reached max tool call iterations ({max_tool_iterations}) during custom Claude streaming."
            )
            final_content = current_stream_part_content
            
        return final_content

    def _process_stream_iteration(
        self,
        client: anthropic.Anthropic,
        current_messages_history: t.List[dict],
        system_prompt: t.Optional[str],
        max_tokens: int,
        stream_function: t.Callable,
        check_connection: t.Callable,
        stream_params: dict
    ) -> dict:
        """Process a single streaming iteration."""
        current_stream_part_content = ""
        stream_stop_reason = None
        
        # Prepare call kwargs
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
                    # Check connection periodically
                    if stream_idx % 5 == 0:
                        if check_connection and not check_connection(**stream_params):
                            raise ConnectionLostError("Connection was lost!")
                    
                    # Accumulate content
                    current_stream_part_content += text_chunk
                    if text_chunk:
                        stream_function(text_chunk, **stream_params)
                        
                # Get final message details
                final_message_status = stream_handler.get_final_message()
                stream_stop_reason = final_message_status.stop_reason
                logger.debug(f"Claude stream part finished. Stop Reason: {stream_stop_reason}")
                
        except ConnectionLostError:
            # Re-raise immediately
            raise
        except anthropic.APIError as e:
            logger.exception("[CLAUDEAI] API Error during stream processing", exc_info=e)
            raise RetryableCustomError(f"Claude AI API Error: {e}") from e
        except Exception as e:
            logger.exception("Exception during Claude stream processing", exc_info=e)
            raise RetryableCustomError(f"Claude AI stream processing failed: {e}") from e
            
        logger.debug(f"Accumulated stream content for parsing: {current_stream_part_content[:500]}...")
        
        return {
            "content": current_stream_part_content,
            "stop_reason": stream_stop_reason
        }

    def _handle_standard_request(
        self,
        client: anthropic.Anthropic,
        current_messages_history: t.List[dict],
        system_prompt: t.Optional[str],
        max_tokens: int,
        tool_registry: t.Dict[str, ToolDefinition],
        max_tool_iterations: int
    ) -> str:
        """Handle standard (non-streaming) requests to Claude AI."""
        iteration_count = 0
        final_content = ""
        
        while iteration_count < max_tool_iterations:
            iteration_count += 1
            
            # Prepare call kwargs
            call_kwargs = {
                "model": self.model,
                "messages": current_messages_history,
                "max_tokens": max_tokens,
            }
            
            if system_prompt:
                call_kwargs['system'] = system_prompt
                
            logger.debug(f"Calling Claude API with messages: {current_messages_history}")
            
            # Make API call
            response = client.messages.create(**call_kwargs)
            response_text = response.content[0].text if response.content else ""
            
            # Check for tool calls
            parsed_tool_call = parse_tool_call_block(response_text)
            
            if parsed_tool_call:
                # Add assistant message to history
                if response:
                    assistant_message = {"role": response.role, "content": response.content}
                else:
                    assistant_message = {"role": "assistant", "content": response_text}
                current_messages_history.append(assistant_message)
                
                # Handle tool call
                self._handle_tool_call(
                    tool_call=parsed_tool_call,
                    tool_registry=tool_registry,
                    messages_history=current_messages_history
                )
                continue
            else:
                # No tool call - we're done
                text_blocks = [block.text for block in response.content if block.type == "text"]
                final_response_content = "\n".join(text_blocks).strip()
                final_content = final_response_content
                break
                
        return final_content

    def _handle_tool_call(
        self,
        tool_call: dict,
        tool_registry: t.Dict[str, ToolDefinition],
        messages_history: t.List[dict]
    ) -> None:
        """Handle a tool call by executing the tool and updating message history."""
        tool_name = tool_call.get("tool_name")
        parameters = tool_call.get("parameters", {})
        
        # Execute the tool
        tool_result_str = self._execute_tool(tool_name, parameters, tool_registry)
        
        # Add the tool result to history
        tool_result_message = format_tool_result_message(tool_name, tool_result_str)
        messages_history.append(tool_result_message)

    def _execute_tool(
        self,
        tool_name: str,
        parameters: dict,
        tool_registry: t.Dict[str, ToolDefinition]
    ) -> str:
        """Execute a tool and return the result as a string."""
        tool_definition = tool_registry.get(tool_name)
        
        if tool_definition and tool_definition.execution_function:
            try:
                logger.info(f"Executing tool '{tool_name}' with parameters: {parameters}")
                tool_result_str = tool_definition.execution_function(**parameters)
                logger.info(f"Tool '{tool_name}' executed. Result snippet: {tool_result_str[:200]}...")
                return tool_result_str
            except Exception as exec_err:
                logger.exception(f"Error executing tool '{tool_name}'", exc_info=exec_err)
                return json.dumps({"error": f"Failed to execute tool '{tool_name}': {str(exec_err)}"})
        else:
            logger.warning(f"Tool '{tool_name}' requested but not found in registry or not executable.")
            return json.dumps({"error": f"Tool '{tool_name}' is not available."})

    def _create_response(self, content: str, max_tokens: int, kwargs: dict) -> ClaudeAIReponse:
        """Create a ClaudeAIResponse object from the content."""
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