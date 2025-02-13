from lamoom.responses import AIResponse
from dataclasses import dataclass
from openai.types.chat import ChatCompletionMessage as Message


@dataclass(kw_only=True)
class ClaudeAIReponse(AIResponse):
    message: Message = None

    def get_message_str(self) -> str:
        return self.message.model_dump_json(indent=2)

    def __str__(self) -> str:
        result = (
            f"finish_reason: {self.finish_reason}\n"
            f"message: {self.get_message_str()}\n"
        )
        return result
