import logging
import typing as t
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from openai import OpenAI

from lamoom.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
from lamoom.ai_models.constants import C_128K, C_16K, C_32K, C_4K
from lamoom.ai_models.openai.responses import OpenAIResponse
from lamoom.ai_models.utils import get_common_args
from lamoom.exceptions import ConnectionLostError
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
        functions: t.List[t.Dict[str, str]] = [],
        tool_registry: t.Dict[str, ToolDefinition] = {},
        mcp_call_registry: t.Dict[str, t.Callable] = None,
        max_tool_iterations: int = 5,   # Safety limit for sequential calls
        stream_function: t.Callable = None,
        check_connection: t.Callable = None,
        stream_params: dict = {},
        client_secrets: dict = {},
        **kwargs,
    ) -> OpenAIResponse:
                
        client = self.get_client(client_secrets)
        tool_definitions = list(tool_registry.values())
            
        # Inject Tool Prompts into initial messages
        current_messages_history = inject_tool_prompts(messages, tool_definitions)
        
        iteration_count = 0
        while iteration_count < max_tool_iterations:
            iteration_count += 1
            
            logger.info(f"--- Generic Tool Call Iteration: {iteration_count} ---")
            
            call_kwargs = {
                **{
                    "messages": current_messages_history,
                },
                **self.get_params(),
                **kwargs,
            }
            
            try:
                result = client.chat.completions.create(
                    **call_kwargs,
                )
                return OpenAIStreamResponse(
                    stream_function=stream_function,
                    check_connection=check_connection,
                    stream_params=stream_params,
                    original_result=result,
                    client=client,
                    initial_call_kwargs=call_kwargs,
                    tool_registry=tool_registry,
                    mcp_call_registry=mcp_call_registry,
                    max_tool_iterations=max_tool_iterations,
                    initial_messages_history=current_messages_history,
                    prompt=Prompt(
                        messages=kwargs.get("messages"),
                        functions=kwargs.get("tools"),
                        max_tokens=max_tokens,
                        temperature=kwargs.get("temperature"),
                        top_p=kwargs.get("top_p"),
                    ),
                ).stream()
            except Exception as e:
                logger.exception("[OPENAI] failed to handle chat stream", exc_info=e)
                raise_openai_exception(e)


@dataclass(kw_only=True)
class OpenAIStreamResponse(OpenAIResponse):
    stream_function: t.Callable
    check_connection: t.Callable
    stream_params: dict
    
    client: OpenAI
    tool_registry: t.Dict[str, ToolDefinition]
    mcp_call_registry: t.Dict[str, t.Callable]
    max_tool_iterations: int
    initial_call_kwargs: dict
    initial_messages_history: t.List[dict]
    tool_registry: t.Dict[str, t.Callable]

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
            for chunk in stream_iterator:
                stream_idx += 1
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                stream_stop_reason = finish_reason

                if delta and delta.content:
                    text_chunk = delta.content
                    current_stream_part_content += text_chunk
                    self.process_message(text_chunk, stream_idx)

                if finish_reason:
                    logger.debug(f"Stream part finished with reason: {finish_reason}")
                    break

            # Check for tool call after accumulating content
            detected_tool_call = parse_tool_call_block(current_stream_part_content)
            if detected_tool_call:
                logger.info(f'Found tool call in the response: {detected_tool_call}')

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