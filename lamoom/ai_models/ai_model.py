import typing as t
from dataclasses import dataclass, field
from enum import Enum
import logging
from _decimal import Decimal

from lamoom.ai_models.tools.base_tool import ToolDefinition, inject_tool_prompts, parse_tool_call_block
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
        
        # Prepare streaming response
        stream_response = StreamingResponse(
            tool_registry=tool_registry,
            messages=messages
        )
        
        # Inject tool prompts into first message
        current_messages = inject_tool_prompts(messages, tool_definitions)
        
        attempts = max_tool_iterations
        while attempts > 0:
            try:
                stream_response = self.streaming(
                    client=client,
                    messages=current_messages,
                    max_tokens=max_tokens,
                    stream_function=stream_function,
                    check_connection=check_connection,
                    stream_params=stream_params,
                    stream_response=stream_response,
                    **kwargs
                )
                
                if stream_response.is_detected_tool_call:
                    parsed_tool_call = parse_tool_call_block(stream_response.detected_tool_call)
                    if not parsed_tool_call:
                        continue
                        
                    # Execute tool call
                    tool_result = self.handle_tool_call(parsed_tool_call, tool_registry)
                    
                    # Add messages to history
                    stream_response.add_message("assistant", stream_response.content)
                    stream_response.add_tool_result(parsed_tool_call, tool_result)
                    
                    # Update messages for next iteration
                    current_messages = stream_response.messages
                    attempts -= 1
                    continue
                    
                break
                
            except RetryableCustomError:
                attempts -= 1
                continue
                
        return stream_response

    def handle_tool_call(self, tool_call: dict, tool_registry: t.Dict[str, ToolDefinition]) -> str:
        """Handle a tool call by executing the corresponding function from the registry."""
        tool_name = tool_call.get("tool_name")
        parameters = tool_call.get("parameters", {})
        
        tool_function = tool_registry.get(tool_name)
        if not tool_function:
            logger.warning(f"Tool '{tool_name}' not found in registry")
            return json.dumps({"error": f"Tool '{tool_name}' is not available."})
            
        try:
            logger.info(f"Executing tool '{tool_name}' with parameters: {parameters}")
            result = tool_function.execution_function(**parameters)
            logger.info(f"Tool '{tool_name}' executed successfully")
            return json.dumps({"result": result})
        except Exception as e:
            logger.exception(f"Error executing tool '{tool_name}'", exc_info=e)
            return json.dumps({"error": f"Failed to execute tool '{tool_name}': {str(e)}"})

    def streaming(
        self,
        client: t.Any,
        messages: t.List[dict],
        max_tokens: int,
        stream_function: t.Callable,
        check_connection: t.Callable,
        stream_params: dict,
        stream_response: StreamingResponse,
        **kwargs
    ) -> StreamingResponse:
        """Process streaming response. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement streaming method")

    def get_client(self, client_secrets: dict = {}) -> t.Any:
        """Get the client instance. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_client method")


  