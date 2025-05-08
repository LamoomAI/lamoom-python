import typing as t
from dataclasses import dataclass, field
from enum import Enum

from _decimal import Decimal

from lamoom.ai_models.tools.base_tool import ToolDefinition
from lamoom.responses import AIResponse

from logging import getLogger

logger = getLogger(__name__)

class AI_MODELS_PROVIDER(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    CLAUDE = "claude"
    GEMINI = "gemini"
    NEBIUS = "nebius"
    CUSTOM = "custom"


@dataclass(kw_only=True)
class AIModel:
    tiktoken_encoding: t.Optional[str] = "cl100k_base"
    provider: AI_MODELS_PROVIDER = None
    support_functions: bool = False

    @property
    def name(self) -> str:
        return "undefined_aimodel"

    def _decimal(self, value) -> Decimal:
        return Decimal(value).quantize(Decimal(".00001"))

    def get_params(self) -> t.Dict[str, t.Any]:
        return {}

    def call(self, *args, **kwargs) -> AIResponse:
        raise NotImplementedError

    def get_metrics_data(self):
        return {}


    

@dataclass(kw_only=True)
class StreamResponse(AIResponse):
    stream_function: t.Callable
    check_connection: t.Callable
    stream_params: dict
    
    client: AIModel
    tool_registry: t.Dict[str, ToolDefinition]
    max_tool_iterations: int
    initial_call_kwargs: dict
    initial_messages_history: t.List[dict]
    first_response_to_user_ts: t.Optional[int] = None
    exception: t.Optional[Exception] = None

    # Internal state for the stream method
    _current_messages_history: t.List[dict] = field(init=False, default_factory=list)
    _total_accumulated_content: str = field(init=False, default="")
    
    def __post_init__(self):
        self._current_messages_history = list(self.initial_messages_history)

    def process_message(self, text: str, idx: int):
        if idx % 5 == 0:
            if not self.check_connection(**self.stream_params):
                raise ConnectionLostError("Connection was lost!")
        if not text:
            return
        self.stream_function(text, **self.stream_params)

    def _handle_tool_call(self, tool_name: str, parameters: dict) -> str:
        """Handle a tool call by executing the corresponding function from the MCP registry."""
        tool_function = self.mcp_call_registry.get(tool_name)
        if not tool_function:
            logger.warning(f"Tool '{tool_name}' not found in MCP registry")
            return json.dumps({"error": f"Tool '{tool_name}' is not available."})

        try:
            logger.info(f"Executing MCP tool '{tool_name}' with parameters: {parameters}")
            result = tool_function(**parameters)
            logger.info(f"MCP tool '{tool_name}' executed successfully")
            return json.dumps({"result": result})
        except Exception as e:
            logger.exception(f"Error executing MCP tool '{tool_name}'", exc_info=e)
            return json.dumps({"error": f"Failed to execute tool '{tool_name}': {str(e)}"})

    def _process_stream_response(self, stream_iterator) -> t.Tuple[str, str, t.Optional[dict]]:
        """Process a single streaming response from OpenAI.
        
        Returns:
            Tuple of (content, stop_reason, detected_tool_call)
        """
        current_stream_part_content = ""
        stream_stop_reason = None
        detected_tool_call = None
        stream_idx = 0

        try:
            logger.debug("Processing stream chunks...")
            content = ""
            for i, data in enumerate(self.original_result):
                if not data.choices:
                    continue
                msg = data.choices[0]
                delta = msg.delta
                if delta:
                    content += delta.content or ""
                    self.process_message(delta.content, i)

                if '</tool_call>' in delta:
                    detected_tool_call = parse_tool_call_block(delta.content)
                    if detected_tool_call:
                        logger.info(f"Detected tool call: {detected_tool_call}")
                        break

                if 'content' in delta:
                    if not first_response_to_user_ts:
                        first_response_to_user_ts = curr_timestamp_in_ms()
                        chat_stream_response.first_response_to_user_ts = (
                            first_response_to_user_ts
                        )
                    generated_text = delta.content
                    chat_stream_response.content += generated_text or ''
                    process_messages(generated_text, i)

                if message.choices and 'finish_reason' in message.choices[0]:
                    finish_reason = message.choices[0].finish_reason
                    chat_stream_response.message = {
                        'role': 'assistant',
                        'content': chat_stream_response.content,
                    }
                    chat_stream_response.finish_reason = finish_reason
                    return chat_stream_response

        except ConnectionLostError:
            raise
        except Exception as e:
            logger.exception("Exception during stream processing", exc_info=e)
            raise RetryableCustomError(f"OpenAI stream processing failed: {e}") from e

        return current_stream_part_content, stream_stop_reason, detected_tool_call

    def stream(self) -> t.Self:
        """Process the stream, handle tool calls, and manage conversation history."""
        iteration_count = 0
        current_stream_iterator = self.original_result

        while iteration_count < self.max_tool_iterations:
            iteration_count += 1
            logger.info(f"--- OpenAI Tool Streaming Iteration: {iteration_count} ---")

            try:
                current_stream_part_content, stream_stop_reason, detected_tool_call = self._process_stream_response(
                    current_stream_iterator
                )

                # Add the raw assistant response to history
                assistant_message = {"role": "assistant", "content": current_stream_part_content}
                self._current_messages_history.append(assistant_message)

                if detected_tool_call:
                    tool_name = detected_tool_call.get("tool_name")
                    parameters = detected_tool_call.get("parameters", {})
                    
                    # Execute the tool and get result
                    tool_result_str = self._handle_tool_call(tool_name, parameters)
                    
                    # Add tool result to history
                    tool_result_message = format_tool_result_message(tool_name, tool_result_str)
                    self._current_messages_history.append(tool_result_message)

                    # Make a new streaming call with updated history
                    logger.info("Restarting stream after tool execution")
                    new_call_kwargs = {
                        **self.initial_call_kwargs,
                        "messages": self._current_messages_history,
                        "stream": True,
                    }
                    current_stream_iterator = self.client.chat.completions.create(**new_call_kwargs)
                    continue

                # No tool call found - finish streaming
                self._total_accumulated_content = current_stream_part_content
                self.finish_reason = stream_stop_reason or "stop"
                break

            except ConnectionLostError as cle:
                logger.error("Connection lost during stream processing", exc_info=cle)
                self.finish_reason = "error_connection_lost"
                raise cle
            except Exception as e:
                logger.exception("Exception during stream processing", exc_info=e)
                self.finish_reason = "error_processing_stream"
                raise_openai_exception(e)

        if iteration_count >= self.max_tool_iterations:
            logger.warning(f"Reached maximum tool call iterations ({self.max_tool_iterations})")
            self.finish_reason = "error_max_tool_iterations"

        # Set final response fields
        self.content = self._total_accumulated_content
        self.message = Message(
            content=self.content,
            role="assistant"
        )

        logger.debug(f"Stream processing complete. Final finish reason: {self.finish_reason}")
        return self