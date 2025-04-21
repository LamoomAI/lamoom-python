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
from lamoom.ai_models.tools import ToolDefinition, AVAILABLE_TOOLS_REGISTRY, inject_tool_prompts, parse_tool_call_block, \
    format_tool_result_message
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
        tool_registry: t.Dict[str, ToolDefinition] = AVAILABLE_TOOLS_REGISTRY,
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

                #TODO: handle streaming part later
                if kwargs.get("stream"):
                    
                    return OpenAIStreamResponse(
                        stream_function=stream_function,
                        check_connection=check_connection,
                        stream_params=stream_params,
                        original_result=result,
                        client=client,
                        initial_call_kwargs=call_kwargs,
                        tool_registry=tool_registry,
                        initial_messages_history=current_messages_history,
                        max_tool_iterations=max_tool_iterations,
                        prompt=Prompt(
                            messages=kwargs.get("messages"),
                            functions=kwargs.get("tools"),
                            max_tokens=max_tokens,
                            temperature=kwargs.get("temperature"),
                            top_p=kwargs.get("top_p"),
                        ),
                    ).stream()
                    
                logger.debug(f"Result: {result.choices[0]}")
                choice = result.choices[0]
                response_message = choice.message
                # *** TOOL CALL CHECK ***
                response_text = response_message.content
                
                # Parse the response for the <tool_call> block
                parsed_tool_call = parse_tool_call_block(response_text)
                
                if parsed_tool_call:
                    tool_name = parsed_tool_call.get("tool_name")
                    parameters = parsed_tool_call.get("parameters", {})

                    # Add the assistant's message (containing the tool call) to history
                    if response_message:
                        assistant_message_to_add = response_message.__dict__ 
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

                    tool_result_message = format_tool_result_message(tool_name, tool_result_str)
                    current_messages_history.append(tool_result_message)

                    continue

                # *** NO TOOL CALL - Normal Response ***
                else:
                    logger.info("No tool call block found in response. Finishing.")
                    return OpenAIResponse(
                        finish_reason=choice.finish_reason,
                        message=response_message,
                        content=response_message.content,
                        original_result=result,
                        prompt=Prompt(
                            messages=messages, # Original messages
                            functions=call_kwargs.get("tools"),
                            max_tokens=max_tokens,
                            temperature=kwargs.get("temperature"),
                            top_p=kwargs.get("top_p"),
                        ),
                    )
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
    max_tool_iterations: int
    initial_call_kwargs: dict # Original non-message kwargs
    initial_messages_history: t.List[dict] # Original messages list *with tool prompts injected*

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

    def stream(self):
        """
        Processes the stream, parses for custom <tool_call> blocks,
        executes tools, and restarts the stream if necessary.
        """
        iteration_count = 0
        current_stream_iterator = self.original_result # Start with the initial iterator

        while iteration_count < self.max_tool_iterations:
            iteration_count += 1
            logger.info(f"--- Custom Tool Streaming Iteration: {iteration_count} ---")

            current_stream_part_content = ""
            current_finish_reason = None
            assistant_response_for_history = {"role": "assistant", "content": ""} # To store raw assistant text

            stream_idx = 0
            try:
                logger.debug("Processing stream chunks...")
                for chunk in current_stream_iterator:
                    stream_idx += 1
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason
                    current_finish_reason = finish_reason # Track the latest finish reason

                    # Accumulate content and stream out using process_message
                    if delta and delta.content:
                        text_chunk = delta.content
                        current_stream_part_content += text_chunk
                        self.process_message(text_chunk, stream_idx)

                    # If stream part finished, break inner loop
                    if finish_reason:
                        logger.debug(f"Stream part finished with reason: {finish_reason}")
                        break

                # --- After processing chunks for this stream part ---
                logger.debug(f"Accumulated content for parsing: {current_stream_part_content[:500]}...")
                # Update the assistant message content for history
                assistant_response_for_history["content"] = current_stream_part_content
                # Add the raw assistant response to our internal history *before* checking for tool call
                self._current_messages_history.append(assistant_response_for_history)

                # Parse the *accumulated* text for the custom tool block
                parsed_tool_call = parse_tool_call_block(current_stream_part_content)

                if parsed_tool_call:
                    tool_name = parsed_tool_call.get("tool_name")
                    parameters = parsed_tool_call.get("parameters", {})
                    logger.info(f"Custom tool call block parsed: {tool_name}")

                    tool_definition = self.tool_registry.get(tool_name)
                    tool_result_str = "" 

                    # Execute the tool
                    if tool_definition and tool_definition.execution_function:
                        try:
                            logger.info(f"Executing tool '{tool_name}' with parameters: {parameters}")
                            # *** EXECUTE TOOL ***
                            tool_result_str = tool_definition.execution_function(**parameters)
                            logger.info(f"Tool '{tool_name}' executed. Result snippet: {tool_result_str[:200]}...")
                        except Exception as exec_err:
                            logger.exception(f"Error executing tool '{tool_name}'", exc_info=exec_err)
                            tool_result_str = json.dumps({"error": f"Failed to execute tool '{tool_name}': {str(exec_err)}"})
                    else:
                        logger.warning(f"Tool '{tool_name}' requested but not found in registry or not executable.")
                        tool_result_str = json.dumps({"error": f"Tool '{tool_name}' is not available."})

                    tool_result_message = format_tool_result_message(tool_name, tool_result_str)
                    self._current_messages_history.append(tool_result_message)

                    # --- Make a *new* streaming call ---
                    logger.info("Restarting stream after custom tool execution.")
                    new_call_kwargs = {
                        **self.initial_call_kwargs,
                        "messages": self._current_messages_history, 
                        "stream": True, 
                    }

                    current_stream_iterator = self.client.chat.completions.create(**new_call_kwargs)
                    continue

                # --- No Tool Call Found in this stream part ---
                else:
                    logger.info("No custom tool call block found in this stream part.")
                    self._total_accumulated_content = current_stream_part_content # Store final content
                    self.finish_reason = current_finish_reason or "stop"
                    break 

            # --- Exception Handling for the inner loop ---
            except ConnectionLostError as cle:
                 logger.error("Connection lost during stream processing.", exc_info=cle)
                 self.finish_reason = "error_connection_lost"
                 raise cle
            except Exception as e:
                logger.exception("Exception during custom stream chunk processing", exc_info=e)
                self.finish_reason = "error_processing_stream"
                raise_openai_exception(e)

        # --- End of outer while loop ---
        if iteration_count >= self.max_tool_iterations:
             logger.warning(f"Reached maximum tool call iterations ({self.max_tool_iterations}) during custom streaming.")
             self.finish_reason = "error_max_tool_iterations"
             self._total_accumulated_content = current_stream_part_content

        # Populate final fields of the response object
        self.content = self._total_accumulated_content
        self.message = Message( # Use your Message class structure
             content=self.content,
             role="assistant"
             )

        logger.debug(f"Custom stream processing complete. Final finish reason: {self.finish_reason}")
        return self