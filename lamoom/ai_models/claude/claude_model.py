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
            max_tool_iterations: int = 5,   # Safety limit for sequential calls
            **kwargs) -> AIResponse:
        max_tokens = min(max_tokens, self.max_tokens)
        
        common_args = get_common_args(max_tokens)
        kwargs = {
            **common_args,
            **self.get_params(),
            **kwargs,
        }
        messages = self.uny_all_messages_with_same_role(messages)
        
        tool_definitions = list(tool_registry.values())
            
        # Inject Tool Prompts into initial messages
        current_messages_history = inject_tool_prompts(messages, tool_definitions)
        
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

                    current_stream_part_content = ""
                    stream_stop_reason = None # Reason for this specific stream part

                    call_kwargs = {
                            "model": self.model,
                            "max_tokens": max_tokens,
                            "messages": current_messages_history, # Use current history
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
                                    if not check_connection(**stream_params):
                                        raise ConnectionLostError("Connection was lost!")
                                # Accumulate content
                                current_stream_part_content += text_chunk
                                if text_chunk:
                                    stream_function(text_chunk, **stream_params)

                            # Get final message details after stream ends
                            final_message_status = stream_handler.get_final_message()
                            stream_stop_reason = final_message_status.stop_reason
                            logger.debug(f"Claude stream part finished. Stop Reason: {stream_stop_reason}")

                    except ConnectionLostError: # Catch specifically
                         raise # Re-raise immediately
                    except anthropic.APIError as e:
                         logger.exception("[CLAUDEAI] API Error during stream processing", exc_info=e)
                         raise RetryableCustomError(f"Claude AI API Error: {e}") from e
                    except Exception as e:
                        logger.exception("Exception during Claude stream processing", exc_info=e)
                        raise RetryableCustomError(f"Claude AI stream processing failed: {e}") from e

                    # --- After processing stream part ---
                    logger.debug(f"Accumulated stream content for parsing: {current_stream_part_content[:500]}...")

                    # Add the raw assistant response to history *before* parsing
                    assistant_message_to_add = {"role": "assistant", "content": current_stream_part_content}
                    current_messages_history.append(assistant_message_to_add)

                    # Parse the accumulated text for the custom tool block
                    parsed_tool_call = parse_tool_call_block(current_stream_part_content)

                    if parsed_tool_call:
                        tool_name = parsed_tool_call.get("tool_name")
                        parameters = parsed_tool_call.get("parameters", {})
                        logger.info(f"Custom tool call block parsed: {tool_name}")

                        tool_definition = tool_registry.get(tool_name)
                        tool_result_str = "" # Initialize

                        # Execute the tool
                        if tool_definition and tool_definition.execution_function:
                            try:
                                logger.info(f"Executing tool '{tool_name}' with parameters: {parameters}")
                                tool_result_str = tool_definition.execution_function(**parameters)
                                logger.info(f"Tool '{tool_name}' executed. Result snippet: {tool_result_str[:200]}...")
                            except Exception as exec_err:
                                logger.exception(f"Error executing tool '{tool_name}'", exc_info=exec_err)
                                tool_result_str = json.dumps({"error": f"Failed to execute tool '{tool_name}': {str(exec_err)}"})
                        else:
                            logger.warning(f"Tool '{tool_name}' requested but not found/executable.")
                            tool_result_str = json.dumps({"error": f"Tool '{tool_name}' is not available."})

                        tool_result_message = format_tool_result_message(tool_name, tool_result_str)
                        current_messages_history.append(tool_result_message)

                        continue

                    # --- No Tool Call Found in this stream part ---
                    else:
                        logger.info("No custom tool call block found in this stream part. Finishing.")
                        content = current_stream_part_content # Final content is from this last part
                        break

                # --- End of streaming while loop ---
                if iteration_count >= max_tool_iterations:
                    logger.warning(f"Reached max tool call iterations ({max_tool_iterations}) during custom Claude streaming.")
                    # Use content from the last attempt as final content
                    content = current_stream_part_content
            else:
                iteration_count = 0
                while iteration_count < max_tool_iterations:
                    call_kwargs = {
                        "model": self.model,
                        "messages": current_messages_history,
                        "max_tokens": max_tokens,
                    }
                    
                    if system_prompt:
                        call_kwargs['system'] = system_prompt
    
                    logger.debug(f"Calling Claude API with messages: {current_messages_history}")
                        
                    response = client.messages.create(**call_kwargs)
                    # *** TOOL CALL CHECK ***
                    response_text = response.content[0].text if response.content else ""
                    
                    # Parse the response for the <tool_call> block
                    parsed_tool_call = parse_tool_call_block(response_text)
                    if parsed_tool_call:
                        tool_name = parsed_tool_call.get("tool_name")
                        parameters = parsed_tool_call.get("parameters", {})

                        if response:
                            assistant_message_to_add = {"role": response.role, "content": response.content} 
                        else:
                            assistant_message_to_add = {"role": "assistant", "content": response_text}
                        current_messages_history.append(assistant_message_to_add)

                        tool_definition = tool_registry.get(tool_name)

                        # Execute the tool
                        if tool_definition and tool_definition.execution_function:
                            try:
                                logger.info(f"Executing tool '{tool_name}' with parameters: {parameters}")
                                tool_result_str = tool_definition.execution_function(**parameters)
                                logger.info(f"Tool '{tool_name}' executed. Result snippet: {tool_result_str[:200]}...")
                            except Exception as exec_err:
                                logger.exception(f"Error executing tool '{tool_name}'", exc_info=exec_err)
                                tool_result_str = json.dumps({"error": f"Failed to execute tool '{tool_name}': {str(exec_err)}"})
                        else:
                            logger.warning(f"Tool '{tool_name}' requested but not found in registry or not executable.")
                            tool_result_str = json.dumps({"error": f"Tool '{tool_name}' is not available."})

                        # Add the tool result to history
                        tool_result_message = format_tool_result_message(tool_name, tool_result_str)
                        current_messages_history.append(tool_result_message)

                        continue                    
                    else:
                        text_blocks = [block.text for block in response.content if block.type == "text"]
                        final_response_content = "\n".join(text_blocks).strip()
                        content = final_response_content
                        break
                    
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
