import logging
import typing as t
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from openai import OpenAI

from flow_prompt.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
from flow_prompt.ai_models.constants import C_128K, C_16K, C_32K, C_4K
from flow_prompt.ai_models.openai.responses import OpenAIResponse
from flow_prompt.ai_models.ai_model_registry import AIModelRegistry
from flow_prompt.ai_models.utils import get_common_args
from flow_prompt.exceptions import ConnectionLostError

from openai.types.chat import ChatCompletionMessage as Message
from flow_prompt.responses import Prompt

from .utils import raise_openai_exception

M_DAVINCI = "davinci"

logger = logging.getLogger(__name__)


class FamilyModel(Enum):
    chat = "GPT-3.5"
    gpt4 = "GPT-4"
    gpt4o = "GPT-4o"
    gpt4o_mini = "GPT-4o-mini"
    instruct_gpt = "InstructGPT"


DEFAULT_PRICING = {
    "price_per_prompt_1k_tokens": Decimal(0.01),
    "price_per_sample_1k_tokens": Decimal(0.03),
}


OPEN_AI_PRICING = {
    FamilyModel.chat.value: {
        C_4K: {
            "price_per_prompt_1k_tokens": Decimal(0.0015),
            "price_per_sample_1k_tokens": Decimal(0.0020),
        },
        C_16K: {
            "price_per_prompt_1k_tokens": Decimal(0.0010),
            "price_per_sample_1k_tokens": Decimal(0.0020),
        },
        C_128K: {
            "price_per_prompt_1k_tokens": Decimal(0.01),
            "price_per_sample_1k_tokens": Decimal(0.03),
        },
    },
    FamilyModel.gpt4.value: {
        C_4K: {
            "price_per_prompt_1k_tokens": Decimal(0.03),
            "price_per_sample_1k_tokens": Decimal(0.06),
        },
        C_32K: {
            "price_per_prompt_1k_tokens": Decimal(0.06),
            "price_per_sample_1k_tokens": Decimal(0.12),
        },
        C_128K: {
            "price_per_prompt_1k_tokens": Decimal(0.01),
            "price_per_sample_1k_tokens": Decimal(0.03),
        },
    },
    FamilyModel.gpt4o.value: {
        C_128K: {
            "price_per_prompt_1k_tokens": Decimal(0.005),
            "price_per_sample_1k_tokens": Decimal(0.015),
        },
    },
    FamilyModel.gpt4o_mini.value: {
        C_128K: {
            "price_per_prompt_1k_tokens": Decimal(0.00015),
            "price_per_sample_1k_tokens": Decimal(0.0006),
        },
    },
    FamilyModel.instruct_gpt.value: {
        M_DAVINCI: {
            "price_per_prompt_1k_tokens": Decimal(0.0015),
            "price_per_sample_1k_tokens": Decimal(0.002),
        },
    },
}


@AIModelRegistry.register('openai', {'model'})
@dataclass(kw_only=True)
class OpenAIModel(AIModel):
    model: t.Optional[str]
    support_functions: bool = False
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.OPENAI
    family: str = None
    max_sample_budget: int = C_4K

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
            logger.warning(
                f"Unknown family for {self.model}. Please add it obviously. Setting as GPT4"
            )
            self.family = FamilyModel.gpt4.value
        logger.debug(f"Initialized OpenAIModel: {self}")

    @property
    def name(self) -> str:
        return self.model

    @property
    def price_per_prompt_1k_tokens(self) -> Decimal:
        return OPEN_AI_PRICING[self.family].get(self.max_tokens, DEFAULT_PRICING)[
            "price_per_prompt_1k_tokens"
        ]

    @property
    def price_per_sample_1k_tokens(self) -> Decimal:
        return OPEN_AI_PRICING[self.family].get(self.max_tokens, DEFAULT_PRICING)[
            "price_per_sample_1k_tokens"
        ]

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
        }

    def get_metrics_data(self):
        return {
            "model": self.model,
            "family": self.family,
            "provider": self.provider.value,
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
            FamilyModel.gpt4o_mini.value
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
            organization=client_secrets.get("organization"),
            api_key=client_secrets["api_key"],
        )

    def call_chat_completion(
        self,
        messages: t.List[t.Dict[str, str]],
        max_tokens: t.Optional[int],
        functions: t.List[t.Dict[str, str]] = [],
        stream_function: t.Callable = None,
        check_connection: t.Callable = None,
        stream_params: dict = {},
        client_secrets: dict = {},
        **kwargs,
    ) -> OpenAIResponse:
        max_tokens = min(max_tokens, self.max_tokens, self.max_sample_budget)
        common_args = get_common_args(max_tokens)
        kwargs = {
            **{
                "messages": messages,
            },
            **common_args,
            **self.get_params(),
            **kwargs,
        }
        if functions:
            kwargs["tools"] = functions
        try:
            client = self.get_client(client_secrets)
            result = client.chat.completions.create(
                **kwargs,
            )

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
            return OpenAIResponse(
                finish_reason=result.choices[0].finish_reason,
                message=result.choices[0].message,
                content=result.choices[0].message.content,
                original_result=result,
                prompt=Prompt(
                    messages=kwargs.get("messages"),
                    functions=kwargs.get("tools"),
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
