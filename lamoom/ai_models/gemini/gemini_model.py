
from lamoom.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
import logging

from lamoom.ai_models.constants import C_1M, C_128K, C_8K
from lamoom.responses import AIResponse
from decimal import Decimal
from enum import Enum

import typing as t
from dataclasses import dataclass

from lamoom.ai_models.gemini.responses import GeminiAIResponse

from lamoom.ai_models.utils import get_common_args
from openai.types.chat import ChatCompletionMessage as Message
from lamoom.responses import Prompt
from lamoom.exceptions import RetryableCustomError, ConnectionLostError
import google.generativeai as genai


logger = logging.getLogger(__name__)


FLASH = "gemini-1.5-flash"
PRO = "gemini-1.5-pro"
PRO_1_0 = "gemini-1.0-pro"



class FamilyModel(Enum):
    flash = "Gemini 1.5 Flash"
    pro = "Gemini 1.5 Pro"
    pro_1_0 = "Gemini 1.0 Pro"


DEFAULT_PRICING = {
    "price_per_prompt_1k_tokens": Decimal(0.00035),
    "price_per_sample_1k_tokens": Decimal(0.00105),
}

GEMINI_AI_PRICING = {
    FamilyModel.flash.value: {
        C_128K: {
            "price_per_prompt_1k_tokens": Decimal(0.0075),
            "price_per_sample_1k_tokens": Decimal(0.030),
        },
        C_1M: {
            "price_per_prompt_1k_tokens": Decimal(0.015),
            "price_per_sample_1k_tokens": Decimal(0.060),
        }
    },
    FamilyModel.pro_1_0.value: {
        C_1M: {
            "price_per_prompt_1k_tokens": Decimal(0.0005),
            "price_per_sample_1k_tokens": Decimal(0.0015),
        }
    },
    FamilyModel.pro.value: {
        C_128K: {
            "price_per_prompt_1k_tokens": Decimal(0.0035),
            "price_per_sample_1k_tokens": Decimal(0.0105),
        },
        C_1M: {
            "price_per_prompt_1k_tokens": Decimal(0.007),
            "price_per_sample_1k_tokens": Decimal(0.021),
        }
    },
}


@dataclass(kw_only=True)
class GeminiAIModel(AIModel):
    model: str
    max_tokens: int = C_8K
    gemini_model: genai.GenerativeModel = None
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.GEMINI
    family: str = None

    def __post_init__(self):
        self.model = self.model.lower()
        if FLASH in self.model:
            self.family = FamilyModel.flash.value
        elif PRO_1_0 in self.model:
            self.family = FamilyModel.pro_1_0.value
        elif PRO in self.model:
            self.family = FamilyModel.pro.value
        else:
            logger.warning(
                f"Unknown family for {self.model}. Please add it obviously. Setting as Gemini 1.5 Flash"
            )
            self.family = FamilyModel.flash.value

    def call(self, messages: t.List[dict], max_tokens: int, client_secrets: dict = {}, **kwargs) -> AIResponse:
        genai.configure(api_key=client_secrets["api_key"])
        self.gemini_model = genai.GenerativeModel(self.model)
        max_tokens = min(max_tokens, self.max_tokens)
        common_args = get_common_args(max_tokens)
        kwargs = {
            **{
                "messages": messages,
            },
            **common_args,
            **client_secrets,
            **client_secrets,
            **self.get_params(),
            **kwargs,
        }

        logger.debug(
            f"Calling {messages} with max_tokens {max_tokens} and kwargs {kwargs}"
        )

        stream_function = kwargs.get("stream_function")
        check_connection = kwargs.get("check_connection")
        stream_params = kwargs.get("stream_params")

        # Parse only prompt content due to gemini call specifics
        prompt = '\n\n'.join([obj["content"] for obj in messages])

        content = ""

        try:
            if not kwargs.get('stream'):
                response = self.gemini_model.generate_content(prompt, stream=False)
                content = response.text
            else:
                response = self.gemini_model.generate_content(prompt, stream=True)
                idx = 0
                for chunk in response:
                    if idx % 5 == 0:
                        idx = 0
                        if not check_connection(**stream_params):
                            raise ConnectionLostError("Connection was lost!")
                    stream_function(chunk.text, **stream_params)
                    content += chunk.text
                    idx += 1

            return GeminiAIResponse(
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
            logger.exception("[GEMINIAI] failed to handle chat stream", exc_info=e)
            raise RetryableCustomError(f"Gemini AI call failed!")

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
        }
    

    def get_prompt_price(self, count_tokens: int) -> Decimal:
        for key in sorted(GEMINI_AI_PRICING[self.family].keys()):
            if count_tokens < key:
                logger.info(f"Prompt price for {count_tokens} tokens is {GEMINI_AI_PRICING[self.family][key]['price_per_prompt_1k_tokens'] * Decimal(count_tokens) / 1000}")
                return self._decimal(GEMINI_AI_PRICING[self.family][key]["price_per_prompt_1k_tokens"] * Decimal(count_tokens) / 1000)
        
        return self._decimal(self.price_per_prompt_1k_tokens * Decimal(count_tokens) / 1000)
    
    def get_sample_price(self, prompt_sample, count_tokens: int) -> Decimal:
        for key in sorted(GEMINI_AI_PRICING[self.family].keys()):
            if prompt_sample < key:
                logger.info(f"Sample price for {count_tokens} tokens is {GEMINI_AI_PRICING[self.family][key]['price_per_prompt_1k_tokens'] * Decimal(count_tokens) / 1000}")
                return self._decimal(GEMINI_AI_PRICING[self.family][key]["price_per_sample_1k_tokens"] * Decimal(count_tokens) / 1000)
        return self._decimal(self.price_per_sample_1k_tokens * Decimal(count_tokens) / 1000)