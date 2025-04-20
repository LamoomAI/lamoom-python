import logging
import typing as t
from dataclasses import dataclass
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
        tool_definitions = []
        for tool_definition in tool_registry.values():
            tool_definitions.append(tool_definition)
            
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

    def process_message(self, text: str, idx: int):
        if idx % 5 == 0:
            if not self.check_connection(**self.stream_params):
                raise ConnectionLostError("Connection was lost!")
        if not text:
            return
        self.stream_function(text, **self.stream_params)

    def stream(self):
        content = ""
        for i, data in enumerate(self.original_result):
            if not data.choices:
                continue
            choice = data.choices[0]
            if choice.delta:
                content += choice.delta.content or ""
                self.process_message(choice.delta.content, i)
        self.message = Message(
            content=content,
            role="assistant",
        )
        return self