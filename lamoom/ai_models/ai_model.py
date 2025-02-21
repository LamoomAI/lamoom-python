import typing as t
from dataclasses import dataclass
from enum import Enum

from _decimal import Decimal

from lamoom.responses import AIResponse



class AI_MODELS_PROVIDER(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    CLAUDE = "claude"
    GEMINI = "gemini"
    NEBIUS = "nebius"


@dataclass(kw_only=True)
class AIModel:
    tiktoken_encoding: t.Optional[str] = "cl100k_base"
    provider: AI_MODELS_PROVIDER = None
    support_functions: bool = False
    _price_per_prompt_1k_tokens: Decimal = None
    _price_per_sample_1k_tokens: Decimal = None

    @property
    def name(self) -> str:
        return "undefined_aimodel"

    @property
    def price_per_prompt_1k_tokens(self) -> Decimal:
        return self._price_per_prompt_1k_tokens

    def _decimal(self, value) -> Decimal:
        return Decimal(value).quantize(Decimal(".00001"))
    
    def get_prompt_price(self, count_tokens: int) -> Decimal:
        return self._decimal(self.price_per_prompt_1k_tokens * Decimal(count_tokens) / 1000)
    
    def get_sample_price(self, prompt_sample, count_tokens: int) -> Decimal:
        return self._decimal(self.price_per_sample_1k_tokens * Decimal(count_tokens) / 1000)

    @property
    def price_per_sample_1k_tokens(self) -> Decimal:
        return self._price_per_sample_1k_tokens

    def get_params(self) -> t.Dict[str, t.Any]:
        return {}

    def call(self, *args, **kwargs) -> AIResponse:
        raise NotImplementedError

    def get_metrics_data(self):
        return {}
