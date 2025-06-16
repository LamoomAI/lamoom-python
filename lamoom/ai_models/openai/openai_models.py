import logging
import typing as t
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI

from lamoom.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
from lamoom.ai_models.constants import C_128K, C_16K, C_32K, C_4K
from lamoom.ai_models.openai.responses import StreamingResponse
from lamoom.exceptions import ConnectionLostError, RetryableCustomError
from lamoom.ai_models.tools.base_tool import TOOL_CALL_END_TAG, TOOL_CALL_START_TAG

from lamoom.responses import FINISH_REASON_ERROR


M_DAVINCI = "davinci"

logger = logging.getLogger(__name__)


class FamilyModel(Enum):
    chat = "GPT-3.5"
    gpt4 = "GPT-4"
    gpt4o = "o4-mini"
    gpt4o_mini = "o4-mini-mini"
    instruct_gpt = "InstructGPT"


BASE_URL_MAPPING = {
    'gemini': "https://generativelanguage.googleapis.com/v1beta/openai/"
}


@dataclass(kw_only=True)
class OpenAIModel(AIModel):
    max_tokens: int = C_16K
    support_functions: bool = False
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.OPENAI
    family: str = None
    max_sample_budget: int = C_4K
    base_url: str = None
    api_key: str = None

    def __str__(self) -> str:
        return f"openai-{self.model}-{self.family}"

    def __post_init__(self):
        if self.model.startswith("davinci"):
            self.family = FamilyModel.instruct_gpt.value
        elif self.model.startswith("gpt-3"):
            self.family = FamilyModel.chat.value
        elif self.model.startswith("o4-mini-mini"):
            self.family = FamilyModel.gpt4o_mini.value
        elif self.model.startswith("o4-mini"):
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


    def is_provider_openai(self):
        return self.provider == AI_MODELS_PROVIDER.OPENAI
    
    
    def get_metrics_data(self):
        return {
            "model": self.model,
            "family": self.family,
            "provider": self.provider.value if not self.provider.is_custom() else self.provider_name,
            "base_url": self.base_url
        }

    def get_client(self, client_secrets: dict = {}):
        base_url = client_secrets.get("base_url", None)
        return OpenAI(
            organization=client_secrets.get("organization", None),
            api_key=client_secrets["api_key"],
            base_url=client_secrets.get("base_url", None),
        )

    def streaming(
        self,
        client: OpenAI,
        stream_response: StreamingResponse,
        max_tokens: int,
        stream_function: t.Callable,
        check_connection: t.Callable,
        stream_params: dict,
        **kwargs
    ) -> StreamingResponse:
        """Process streaming response from OpenAI."""
        tool_call_started = False
        content = ""

        try:
            call_kwargs = {
                "messages": stream_response.messages,
                "stream": True,
                **self.get_params(),
                **kwargs
            }
            if max_tokens:
                call_kwargs["max_completion_tokens"] = min(max_tokens, self.max_sample_budget)
            logger.info(f"Calling OpenAI with params: {call_kwargs}")
            completion = client.chat.completions.create(**call_kwargs)
            for part in completion:
                if not part.choices:
                    continue
                    
                delta = part.choices[0].delta
                if part.choices and 'finish_reason' in part.choices[0]:
                    logger.info(f'Finish reason: {part.choices[0].finish_reason}')
                    stream_response.set_finish_reason(part.choices[0].finish_reason)

                # Check for tool call markers
                if TOOL_CALL_START_TAG in content and not tool_call_started:
                    tool_call_started = True
                    logger.info(f'tool_call_started: {tool_call_started}')

                if not delta or (not delta.content and getattr(delta, 'reasoning', None)):
                    continue

                if delta.content:
                    content += delta.content
                    if stream_function or self._tag_parser.is_custom_tags():
                        text_to_stream = self.text_to_stream_chunk(delta.content)
                        if text_to_stream:
                            stream_response.streaming_content += text_to_stream
                            if stream_function:
                                stream_function(text_to_stream, **stream_params)
                    
                if getattr(delta, 'reasoning', None) and delta.reasoning:
                    logger.debug(f'Adding reasoning {delta.reasoning}')
                    stream_response.reasoning += delta.reasoning

                if tool_call_started and TOOL_CALL_END_TAG in content:
                    logger.info(f'tool_call_ended: {content}')
                    stream_response.is_detected_tool_call = True
                    stream_response.content = content
                    break

                if check_connection and not check_connection(**stream_params):
                    raise ConnectionLostError("Connection was lost!")

            if stream_function:
                text_to_stream = self.text_to_stream_chunk('')
                if text_to_stream:
                    stream_response.streaming_content += text_to_stream
                    if stream_function:
                        stream_function(text_to_stream, **stream_params)
            stream_response.content = content
            return stream_response
            
        except Exception as e:
            stream_response.content = content
            stream_response.set_finish_reason(FINISH_REASON_ERROR)
            logger.exception("Exception during stream processing", exc_info=e)
            raise RetryableCustomError(f"OpenAI stream processing failed: {e}") from e