import logging
import typing as t
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from openai import OpenAI

from lamoom.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
from lamoom.ai_models.constants import C_128K, C_16K, C_32K, C_4K
from lamoom.ai_models.openai.responses import OpenAIResponse, StreamingResponse
from lamoom.ai_models.utils import get_common_args
from lamoom.exceptions import ConnectionLostError, RetryableCustomError
from lamoom.ai_models.tools.base_tool import ToolDefinition, inject_tool_prompts, parse_tool_call_block
import json

from openai.types.chat import ChatCompletionMessage as Message
from lamoom.responses import Prompt

from .utils import raise_openai_exception

M_DAVINCI = "davinci"

logger = logging.getLogger(__name__)


class FamilyModel(Enum):
    chat = "GPT-3.5"
    gpt4 = "GPT-4"
    gpt4o = "GPT-4o"
    gpt4o_mini = "GPT-4o-mini"
    instruct_gpt = "InstructGPT"

BASE_URL_MAPPING = {
    'gemini': "https://generativelanguage.googleapis.com/v1beta/openai/",
    'nebius': 'https://api.studio.nebius.ai/v1/'
}


@dataclass(kw_only=True)
class OpenAIModel(AIModel):
    model: t.Optional[str]
    max_tokens: int = C_16K
    support_functions: bool = False
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.OPENAI
    family: str = None
    max_sample_budget: int = C_4K
    base_url: str = None

    def __str__(self) -> str:
        return f"openai-{self.model}-{self.family}"

    def __post_init__(self):
        if self.model.startswith("davinci"):
            self.family = FamilyModel.instruct_gpt.value
        elif self.model.startswith("gpt-3"):
            self.family = FamilyModel.chat.value
        elif self.model.startswith("gpt-4o-mini"):
            self.family = FamilyModel.gpt4o_mini.value
        elif self.model.startswith("gpt-4o"):
            self.family = FamilyModel.gpt4o.value
        elif self.model.startswith(("gpt4", "gpt-4", "gpt")):
            self.family = FamilyModel.gpt4.value
        else:
            logger.info(
                f"Unknown family for {self.model}. Please add it obviously. Setting as GPT4"
            )
            self.family = FamilyModel.gpt4.value
        logger.debug(f"Initialized OpenAIModel: {self}")

    @property
    def name(self) -> str:
        return self.model

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
        }

    def get_base_url(self) -> str | None:
        return BASE_URL_MAPPING.get(self.provider.value, None)
    
    def get_metrics_data(self):
        return {
            "model": self.model,
            "family": self.family,
            "provider": self.provider.value,
            "base_url": self.get_base_url() if self.base_url is None else self.base_url
        }

    def call(
        self,
        messages,
        max_tokens,
        stream_function: t.Callable = None,
        check_connection: t.Callable = None,
        stream_params: dict = {},
        client_secrets: dict = {},
        **kwargs,
    ) -> OpenAIResponse:
        logger.debug(
            f"Calling {messages} with max_tokens {max_tokens} and kwargs {kwargs}"
        )
        if self.family in [
            FamilyModel.chat.value,
            FamilyModel.gpt4.value,
            FamilyModel.gpt4o.value,
            FamilyModel.gpt4o_mini.value,
        ]:
            return self.call_chat_completion(
                messages,
                max_tokens,
                stream_function=stream_function,
                check_connection=check_connection,
                stream_params=stream_params,
                client_secrets=client_secrets,
                **kwargs,
            )
        raise NotImplementedError(f"Openai family {self.family} is not implemented")

    def get_client(self, client_secrets: dict = {}):
        return OpenAI(
            organization=client_secrets.get("organization", None),
            api_key=client_secrets["api_key"],
            base_url=self.get_base_url() if self.base_url is None else self.base_url
        )

    def call_chat_completion(
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
    ) -> OpenAIResponse:
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
                stream_response = self._streaming(
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

    def handle_tool_call(self, tool_name: str, parameters: dict, tool_registry: t.Dict[str, ToolDefinition]) -> str:
        """Handle a tool call by executing the corresponding function from the registry."""
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
        client: OpenAI,
        messages: t.List[dict],
        max_tokens: int,
        stream_function: t.Callable,
        check_connection: t.Callable,
        stream_params: dict,
        stream_response: StreamingResponse,
        **kwargs
    ) -> StreamingResponse:
        """Process streaming response from OpenAI."""
        tool_call_started = False
        content = ""
        try:
            call_kwargs = {
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
                **self.get_params(),
                **kwargs
            }
            
            for part in client.chat.completions.create(**call_kwargs):
                if not part.choices:
                    continue
                    
                delta = part.choices[0].delta
                if not delta:
                    continue
                    
                if delta.content:
                    content += delta.content
                    if stream_function:
                        stream_function(delta.content, **stream_params)
                        
                if check_connection and not check_connection(**stream_params):
                    raise ConnectionLostError("Connection was lost!")
                    
                # Check for tool call markers
                if "<tool_call>" in content:
                    if not tool_call_started:
                        tool_call_started = True
                    continue
                    
                if tool_call_started and "</tool_call>" in content:
                    stream_response.is_detected_tool_call = True
                    stream_response.detected_tool_call = content
                    break
            stream_response.content = content
            return stream_response
        except Exception as e:
            stream_response.content = content
            logger.exception("Exception during stream processing", exc_info=e)
            raise RetryableCustomError(f"OpenAI stream processing failed: {e}") from e