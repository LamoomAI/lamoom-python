from flow_prompt.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
import logging

from flow_prompt.ai_models.constants import C_1M
from flow_prompt.ai_models.ai_model_registry import AIModelRegistry
from flow_prompt.responses import AIResponse
from decimal import Decimal

import typing as t
from dataclasses import dataclass

from openai.types.chat import ChatCompletionMessage as Message
from flow_prompt.responses import Prompt
from flow_prompt.exceptions import RetryableCustomError, ConnectionLostError
import openai
from flow_prompt.ai_models.nebius.responses import NebiusAIResponse
from enum import Enum


logger = logging.getLogger(__name__)

FAMILY_MODELS_MAP = {
    'deepseek-ai/DeepSeek-R1': 'DeepSeek R1',
    'deepseek-ai/DeepSeek-V3': 'DeepSeek V3'
}

class FamilyModel(Enum):
    deep_seek_r1 = "DeepSeek R1"
    deep_seek_v3 = "DeepSeek V3"


DEFAULT_PRICING = {
    "price_per_prompt_1k_tokens": Decimal(0.5),
    "price_per_sample_1k_tokens": Decimal(1.5),
}

NEBIUS_AI_PRICING = {
    FamilyModel.deep_seek_r1.value: {
        C_1M: {
            "price_per_prompt_1k_tokens": Decimal(0.8),
            "price_per_sample_1k_tokens": Decimal(2.4),
        }
    },
    FamilyModel.deep_seek_v3.value: {
        C_1M: {
            "price_per_prompt_1k_tokens": Decimal(0.5),
            "price_per_sample_1k_tokens": Decimal(1.5),
        }
    }
}

@AIModelRegistry.register('nebius', {'model'})
@dataclass(kw_only=True)
class NebiusAIModel(AIModel):
    model: str
    max_tokens: int = C_1M
    openai_client: openai.OpenAI = None
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.NEBIUS
    family: str = None

    def __post_init__(self):
        if self.model in FAMILY_MODELS_MAP:
            self.family = FAMILY_MODELS_MAP[self.model]
        else:
            logger.warning(
                f"Unknown family for {self.model}. Defaulting to Nebius DeepSeek R1"
            )
            self.family = FamilyModel.deep_seek_r1.value

    def call(
        self, 
        messages: t.List[dict], 
        max_tokens: int, 
        stream_function: t.Callable = None,
        check_connection: t.Callable = None,
        stream_params: dict = {},
        client_secrets: dict = {}, 
        **kwargs) -> AIResponse:
        
        self.openai_client = openai.OpenAI(
            api_key=client_secrets["api_key"],
            base_url=client_secrets["base_url"]
        )

        common_args = self._get_common_args(max_tokens)
        request_args = {
            "model": self.model,
            "messages": messages,
            **common_args,
            **self.get_params(),
            **kwargs
        }

        logger.debug(
            f"Calling Nebius model {self.model} with max_tokens {max_tokens} "
            f"and args { {k: v for k, v in request_args.items() if k != 'messages'} }"
        )

        try:
            response = self.openai_client.chat.completions.create(**request_args)
            if kwargs.get('stream'):
                return NebiusAIStreamResponse(
                    stream_function=stream_function,
                    check_connection=check_connection,
                    stream_params=stream_params,
                    original_result=response,
                    prompt=Prompt(
                        messages=kwargs.get("messages"),
                        functions=kwargs.get("tools"),
                        max_tokens=max_tokens,
                        temperature=kwargs.get("temperature"),
                        top_p=kwargs.get("top_p"),
                    ),
                ).stream()
                
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            message = response.choices[0].message

            return NebiusAIResponse(
                message=message,
                content=content,
                finish_reason=finish_reason,
                prompt=Prompt(
                    messages=messages,
                    functions=kwargs.get("tools"),
                    max_tokens=max_tokens,
                    temperature=request_args.get("temperature"),
                    top_p=request_args.get("top_p"),
                ),
            )

        except Exception as e:
            logger.exception("[NEBIUSAI] Failed to handle chat completion", exc_info=e)
            raise RetryableCustomError("Nebius AI call failed!") from e

    def name(self) -> str:
        return self.model

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
        }

    def get_metrics_data(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "provider": self.provider.value
        }

    def _get_common_args(self, max_tokens: int) -> dict:
        return {
            "max_tokens": min(max_tokens, self.max_tokens),
            "temperature": 0.7,
            "top_p": 1.0,
        }

    def get_prompt_price(self, count_tokens: int) -> Decimal:
        for tier in sorted(NEBIUS_AI_PRICING[self.family].keys()):
            if count_tokens <= tier:
                price = NEBIUS_AI_PRICING[self.family][tier]["price_per_prompt_1k_tokens"]
                logger.info(f"Prompt price for {count_tokens} tokens is {price}")
                return self._decimal(price * Decimal(count_tokens) / 1000)
        return self._decimal(self.price_per_prompt_1k_tokens * Decimal(count_tokens) / 1000)

    def get_sample_price(self, prompt_sample, count_tokens: int) -> Decimal:
        for tier in sorted(NEBIUS_AI_PRICING[self.family].keys()):
            if prompt_sample <= tier:
                price = NEBIUS_AI_PRICING[self.family][tier]["price_per_sample_1k_tokens"]
                logger.info(f"Sample price for {count_tokens} tokens is {price}")
                return self._decimal(price * Decimal(count_tokens) / 1000)
        return self._decimal(self.price_per_sample_1k_tokens * Decimal(count_tokens) / 1000)
    
    
@dataclass(kw_only=True)
class NebiusAIStreamResponse(NebiusAIResponse):
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