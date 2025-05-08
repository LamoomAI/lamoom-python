import json
import typing as t
from dataclasses import dataclass
from enum import Enum
import logging
from _decimal import Decimal

from lamoom.ai_models.tools.base_tool import ToolCallResult, ToolDefinition, inject_tool_prompts, parse_tool_call_block
from lamoom.responses import AIResponse, StreamingResponse
from lamoom.exceptions import RetryableCustomError

logger = logging.getLogger(__name__)

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

    def get_metrics_data(self):
        return {}

    def call(
        self,
        messages: t.List[t.Dict[str, str]],
        max_tokens: t.Optional[int],
        tool_registry: t.Dict[str, ToolDefinition] = {},
        max_tool_iterations: int = 5,   # Safety limit for sequential calls
        stream_function: t.Callable = None,
        check_connection: t.Callable = None,
        stream_params: dict = {},
        client_secrets: dict = {},
        **kwargs,
    ) -> AIResponse:
        """Common call implementation that handles streaming and tool calls."""
        client = self.get_client(client_secrets)
        tool_definitions = list(tool_registry.values())
        # Inject tool prompts into first message
        current_messages = inject_tool_prompts(messages, tool_definitions)
        # Prepare streaming response
        stream_response = StreamingResponse(
            tool_registry=tool_registry,
            messages=current_messages
        )
        attempts = max_tool_iterations
        while attempts > 0:
            try:
                stream_response = self.streaming(
                    client=client,
                    stream_response=stream_response,
                    max_tokens=max_tokens,
                    stream_function=stream_function,
                    check_connection=check_connection,
                    stream_params=stream_params,
                    **kwargs
                )
                logger.info(f'stream_response: {stream_response}')
                if stream_response.is_detected_tool_call:
                    logger.info(f'is_detected_tool_call')
                    parsed_tool_call = parse_tool_call_block(stream_response.content)
                    logger.info(f'parsed_tool_call {parsed_tool_call}')
                    if not parsed_tool_call:
                        continue
                    # Execute tool call
                    self.handle_tool_call(parsed_tool_call, tool_registry)
                    # Add messages to history
                    logger.info(f'executed parsed_tool_call {parsed_tool_call}')
                    stream_response.add_message("assistant", stream_response.content)
                    stream_response.add_tool_result(parsed_tool_call)
                    attempts -= 1
                    logger.info(f'Left attempts: {attempts}, Added message {stream_response.messages[-1]}')
                    continue
                logger.info(f'Passing execution, finished')
                break
            except RetryableCustomError:
                attempts -= 1
                continue                
        return stream_response


    def handle_tool_call(self, tool_call: ToolCallResult, tool_registry: t.Dict[str, ToolDefinition]) -> str:
        """Handle a tool call by executing the corresponding function from the registry."""
        tool_name = tool_call.tool_name
        parameters = tool_call.parameters
        
        tool_function = tool_registry.get(tool_name)
        if not tool_function:
            logger.warning(f"Tool '{tool_name}' not found in registry")
            return json.dumps({"error": f"Tool '{tool_name}' is not available."})
            
        try:
            logger.info(f"Executing tool '{tool_name}' with parameters: {parameters}")
            result = tool_function.execution_function(**parameters)
            logger.info(f"Tool '{tool_name}' executed successfully")
            tool_call.execution_result = result
            return json.dumps({"result": result})
        except Exception as e:
            result = f"Error executing tool '{tool_name}'"
            logger.exception(result, exc_info=e)
            tool_call.execution_result = result
            return json.dumps({"error": f"{result}: {str(e)}"})

    def streaming(
        self,
        client: t.Any,
        stream_response: StreamingResponse,
        max_tokens: int,
        stream_function: t.Callable,
        check_connection: t.Callable,
        stream_params: dict,
        **kwargs
    ) -> StreamingResponse:
        """Process streaming response. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement streaming method")

    def get_client(self, client_secrets: dict = {}) -> t.Any:
        """Get the client instance. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_client method")
