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
from lamoom.responses import FINISH_REASON_ERROR, Prompt

from .utils import raise_openai_exception

M_DAVINCI = "davinci"

logger = logging.getLogger(__name__)


class FamilyModel(Enum):
    chat = "GPT-3.5"
    gpt4 = "GPT-4"
    gpt4o = "o4-mini"
    gpt4o_mini = "o4-mini-mini"
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

    def get_base_url(self) -> str | None:
        return BASE_URL_MAPPING.get(self.provider.value, None)
    
    def get_metrics_data(self):
        return {
            "model": self.model,
            "family": self.family,
            "provider": self.provider.value,
            "base_url": self.get_base_url() if self.base_url is None else self.base_url
        }

    def get_client(self, client_secrets: dict = {}):
        return OpenAI(
            organization=client_secrets.get("organization", None),
            api_key=client_secrets["api_key"],
            base_url=self.get_base_url() if self.base_url is None else self.base_url
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
            
            for part in client.chat.completions.create(**call_kwargs):
                if not part.choices:
                    continue
                    
                delta = part.choices[0].delta
                if part.choices and 'finish_reason' in part.choices[0]:
                    stream_response.finish_reason = part.choices[0].finish_reason
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
                    stream_response.content = content
                    break
                    
            stream_response.content = content
            return stream_response
            
        except Exception as e:
            stream_response.content = content
            stream_response.finish_reason = FINISH_REASON_ERROR
            logger.exception("Exception during stream processing", exc_info=e)
            raise RetryableCustomError(f"OpenAI stream processing failed: {e}") from e