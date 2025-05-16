
import json
import typing as t
from dataclasses import dataclass
from enum import Enum
import logging
from _decimal import Decimal

import tiktoken

from lamoom import settings
from lamoom.ai_models.tools.base_tool import ToolCallResult, ToolDefinition, parse_tool_call_block
from lamoom.responses import AIResponse, StreamingResponse
from lamoom.exceptions import RetryableCustomError, StopStreamingError
from lamoom.utils import current_timestamp_ms

logger = logging.getLogger(__name__)

class AI_MODELS_PROVIDER(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    CLAUDE = "claude"
    GEMINI = "gemini"
    CUSTOM = "custom"

    def is_custom(self):
        return self == AI_MODELS_PROVIDER.CUSTOM


encoding = tiktoken.get_encoding("cl100k_base")


@dataclass(kw_only=True)
class AIModel:
    model: t.Optional[str] = ''
    tiktoken_encoding: t.Optional[str] = "cl100k_base"
    support_functions: bool = False
    _provider_name: str = None

    @property
    def provider_name(self):
        return self.provider.value if not self.provider.is_custom() else self._provider_name

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
        current_messages: t.List[t.Dict[str, str]],
        max_tokens: t.Optional[int],
        tool_registry: t.Dict[str, ToolDefinition] = {},
        max_tool_iterations: int = 5,   # Safety limit for sequential calls
        stream_function: t.Callable = None,
        check_connection: t.Callable = None,
        stream_params: dict = {},
        client_secrets: dict = {},
        modelname='',
        prompt: 'Prompt' = None,
        context: str = '',
        test_data: dict = {},
        client: t.Any = None,
        **kwargs,
    ) -> AIResponse:
        """Common call implementation that handles streaming and tool calls."""
        model_client = self.get_client(client_secrets)
        # Prepare streaming response
        stream_response = StreamingResponse(
            tool_registry=tool_registry,
            messages=current_messages
        )
        modelname = modelname.replace('/', '_').replace('-', '_')
        attempts = max_tool_iterations
        while attempts > 0:
            try:
                stream_response.update_to_another_attempt()
                stream_response = self.streaming(
                    client=model_client,
                    stream_response=stream_response,
                    max_tokens=max_tokens,
                    stream_function=stream_function,
                    check_connection=check_connection,
                    stream_params=stream_params,
                    **kwargs
                )
                logger.info(f'stream_response: {stream_response}')
                if stream_response.is_detected_tool_call:
                    parsed_tool_call = parse_tool_call_block(stream_response.content)

                    logger.info(f'parsed_tool_call {parsed_tool_call}')
                    if not parsed_tool_call or attempts <= 1:
                        stream_response.add_assistant_message()
                        self.save_call(stream_response, prompt, context, attempt=max_tool_iterations - attempts, client=client)
                        attempts -= 1
                        continue
                    # Execute tool call
                    self.handle_tool_call(parsed_tool_call, tool_registry)
                    # Add messages to history
                    logger.info(f'executed parsed_tool_call {parsed_tool_call}')
                    stream_response.add_tool_result(parsed_tool_call)
                    self.save_call(stream_response, prompt, context, attempt=max_tool_iterations - attempts, client=client)
                    attempts -= 1
                    continue
                stream_response.add_assistant_message()
                self.save_call(stream_response, prompt, context, test_data=test_data, client=client)
                logger.info(f'Passing execution {modelname}, finished. {attempts}')
                break
            except RetryableCustomError as e:
                logger.exception(f'RetryableCustomError {e}')
                attempts -= 1
                continue  
            except StopStreamingError as e:
                logger.exception(f'StopStreamingError {e}')
                stream_response.add_assistant_message()
                self.save_call(stream_response, prompt, context, attempt=max_tool_iterations - attempts, client=client)
                logger.info(f'Passing execution {modelname}, finished. {attempts}')
                break
        return stream_response


    def handle_tool_call(self, tool_call: ToolCallResult, tool_registry: t.Dict[str, ToolDefinition]) -> str:
        """Handle a tool call by executing the corresponding function from the registry."""
        function = tool_call.tool_name
        parameters = tool_call.parameters
        
        tool_function = tool_registry.get(function)
        if not tool_function:
            logger.warning(f"Tool '{function}' not found in registry")
            return json.dumps({"error": f"Tool '{function}' is not available."})
            
        try:
            logger.info(f"Executing tool '{function}' with parameters: {parameters}")
            result = tool_function.execution_function(**parameters)
            logger.info(f"Tool '{function}' executed successfully")
            tool_call.execution_result = result
            return json.dumps({"result": result})
        except Exception as e:
            result = f"Error executing tool '{function}', Please try second time."
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
    

    def calculate_budget_for_text(self, text: str) -> int:
        if not text:
            return 0
        return len(encoding.encode(text))
    
    def save_call(self, stream_response: StreamingResponse, prompt: "Prompt", context: str, attempt: int=0, test_data: dict = {}, client: t.Any = None):

        sample_budget = self.calculate_budget_for_text(
            stream_response.get_message_str()
        )
        stream_response.metrics.sample_tokens_used = sample_budget
        stream_response.metrics.prompt_tokens_used = self.calculate_budget_for_text(
            json.dumps(stream_response.messages)
        )
        stream_response.metrics.ai_model_details = (
            self.get_metrics_data()
        )
        stream_response.metrics.latency = current_timestamp_ms() - stream_response.started_tmst

        if settings.USE_API_SERVICE and client.api_token:
            stream_response.id = f"{prompt.id}#{stream_response.started_tmst}" + (f"#{attempt}" if attempt else "")
            client.worker.add_task(
                client.api_token,
                prompt.service_dump(),
                context,
                stream_response,
                {**test_data, "call_model": self.model}
            )
