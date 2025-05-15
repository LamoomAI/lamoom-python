import typing as t
from dataclasses import dataclass

from lamoom.ai_models.ai_model import AIModel


@dataclass
class AttemptToCall:
    ai_model: AIModel
    weight: int = 1  # from 1 to 100, the higher weight the more often it will be called
    # if you wish to limit functions that can be used, or to turn off calling openai functions for this attempt:
    # [] - if empty list of functions, functions are not supported for that call
    # None - if None, no limitations on functions
    # ['function1', 'function2'] - if list of functions, only those functions will be called
    functions: t.List[str] = None
    attempt_number: int = 1
    api_key: str = None

    def __post_init__(self):
        self.id = (
            f"{self.ai_model.name}"
            f"-n{self.attempt_number}-"
            f"{self.ai_model.provider.value}"
        )

    def __str__(self) -> str:
        return self.id

    def params(self) -> t.Dict[str, t.Any]:
        self.ai_model.get_params()

    def get_functions(self) -> t.List[str]:
        # empty list - functions are not supported
        if not self.ai_model.support_functions:
            return []
        # None - no limitations on functions
        if self.functions is None:
            return None
        return self.functions

    def model_max_tokens(self) -> int:
        return self.ai_model.max_tokens

    def tiktoken_encoding(self) -> str:
        return self.ai_model.tiktoken_encoding
