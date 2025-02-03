from flow_prompt.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
import logging

from flow_prompt.ai_models.constants import C_200K
from flow_prompt.responses import AIResponse
from decimal import Decimal
from enum import Enum

import typing as t
from dataclasses import dataclass

from flow_prompt.ai_models.claude.responses import ClaudeAIReponse
from flow_prompt.ai_models.claude.constants import HAIKU, SONNET, OPUS
from flow_prompt.ai_models.utils import get_common_args
from flow_prompt.ai_models.ai_model_registry import AIModelRegistry

from openai.types.chat import ChatCompletionMessage as Message
from flow_prompt.responses import Prompt
from flow_prompt.exceptions import RetryableCustomError, ConnectionLostError
import anthropic

logger = logging.getLogger(__name__)



class FamilyModel(Enum):
    haiku = "Claude 3 Haiku"
    sonnet = "Claude 3 Sonnet"
    opus = "Claude 3 Opus"


DEFAULT_PRICING = {
    "price_per_prompt_1k_tokens": Decimal(0.003),
    "price_per_sample_1k_tokens": Decimal(0.015),
}

CLAUDE_AI_PRICING = {
    FamilyModel.haiku.value: {
        C_200K: {
            "price_per_prompt_1k_tokens": Decimal(0.00025),
            "price_per_sample_1k_tokens": Decimal(0.00125),
        }
    },
    FamilyModel.sonnet.value: {
        C_200K: {
            "price_per_prompt_1k_tokens": Decimal(0.003),
            "price_per_sample_1k_tokens": Decimal(0.015),
        }
    },
    FamilyModel.opus.value: {
        C_200K: {
            "price_per_prompt_1k_tokens": Decimal(0.015),
            "price_per_sample_1k_tokens": Decimal(0.075),
        }
    },
}


@AIModelRegistry.register('claude', {'model'})
@dataclass(kw_only=True)
class ClaudeAIModel(AIModel):
    model: str
    api_key: str = None
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.CLAUDE
    family: str = None

    def __post_init__(self):
        if HAIKU in self.model:
            self.family = FamilyModel.haiku.value
        elif SONNET in self.model:
            self.family = FamilyModel.sonnet.value
        elif OPUS in self.model:
            self.family = FamilyModel.opus.value
        else:
            logger.warning(
                f"Unknown family for {self.model}. Please add it obviously. Setting as Claude 3 Opus"
            )
            self.family = FamilyModel.opus.value

        logger.debug(f"Initialized ClaudeAIModel: {self}")


    def get_client(self, client_secrets: dict) -> anthropic.Anthropic:
        return anthropic.Anthropic(api_key=client_secrets.get('api_key'))


    def uny_all_messages_with_same_role(self, messages: t.List[dict]) -> t.List[dict]:
        result = []
        last_role = None
        for message in messages:
            if message.get("role") == "system":
                message["role"] = "user"
            if last_role != message.get("role"):
                result.append(message)
                last_role = message.get("role")
            else:
                result[-1]["content"] += message.get("content")
        return result


    def call(self, messages: t.List[dict], max_tokens: int, client_secrets: dict = {}, **kwargs) -> AIResponse:
        common_args = get_common_args(max_tokens)
        kwargs = {
            **common_args,
            **self.get_params(),
            **kwargs,
        }
        messages = self.uny_all_messages_with_same_role(messages)

        logger.debug(
            f"Calling {messages} with max_tokens {max_tokens} and kwargs {kwargs}"
        )
        client = self.get_client(client_secrets)

        stream_function = kwargs.get("stream_function")
        check_connection = kwargs.get("check_connection")
        stream_params = kwargs.get("stream_params")

        content = ""

        try:
            if kwargs.get("stream"):
                with client.messages.stream(
                    model=self.model, max_tokens=max_tokens, messages=messages
                ) as stream:
                    idx = 0
                    for text in stream.text_stream:
                        if idx % 5 == 0:
                            if not check_connection(**stream_params):
                                raise ConnectionLostError("Connection was lost!")

                        stream_function(text, **stream_params)
                        content += text
                        idx += 1
            else:
                response = client.messages.create(
                    model=self.model, max_tokens=max_tokens, messages=messages
                )
                content = response.content[0].text
            return ClaudeAIReponse(
                message=Message(content=content, role="assistant"),
                content=content,
                prompt=Prompt(
                    messages=kwargs.get("messages"),
                    functions=kwargs.get("tools"),
                    max_tokens=max_tokens,
                    temperature=kwargs.get("temperature"),
                    top_p=kwargs.get("top_p"),
                ),
            )
        except Exception as e:
            logger.exception("[CLAUDEAI] failed to handle chat stream", exc_info=e)
            raise RetryableCustomError(f"Claude AI call failed!")

    def name(self) -> str:
        return self.model

    @property
    def price_per_prompt_1k_tokens(self) -> Decimal:
        keys = list(CLAUDE_AI_PRICING[self.family].keys())
        def_pricing = CLAUDE_AI_PRICING[self.family].get(keys[0])
        return CLAUDE_AI_PRICING[self.family].get(self.max_tokens, def_pricing)[
            "price_per_prompt_1k_tokens"
        ]

    @property
    def price_per_sample_1k_tokens(self) -> Decimal:
        keys = list(CLAUDE_AI_PRICING[self.family].keys())
        def_pricing = CLAUDE_AI_PRICING[self.family].get(keys[0])
        return CLAUDE_AI_PRICING[self.family].get(self.max_tokens, def_pricing)[
            "price_per_sample_1k_tokens"
        ]

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
        }

    def get_metrics_data(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
        }
