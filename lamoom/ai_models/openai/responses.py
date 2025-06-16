import json
import logging
import typing as t
from dataclasses import dataclass, field

from openai.types.chat import ChatCompletionMessage as Message
from openai.types.chat import ChatCompletionMessageToolCall as ToolCall

from lamoom.responses import FINISH_REASON_LENGTH, FINISH_REASON_TOOL_CALLS, AIResponse
from lamoom.ai_models.tools.base_tool import ToolDefinition


logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class StreamingResponse(AIResponse):
    is_detected_tool_call: bool = False
    tool_registry: t.Dict[str, ToolDefinition] = field(default_factory=dict)
    messages: t.List[dict] = field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})


@dataclass(kw_only=True)
class OpenAIResponse(AIResponse):
    message: Message = None
    exception: t.Optional[Exception] = None

    @property
    def response(self) -> str:
        return self.content or self.message.content

    def is_function(self) -> bool:
        return self.finish_reason == FINISH_REASON_TOOL_CALLS

    @property
    def tool_calls(self) -> t.List[ToolCall]:
        return self.message.tool_calls

    def get_function_name(self, tool_call: ToolCall) -> t.Optional[str]:
        if tool_call.type != "function":
            logger.error(f"function.type is not function: {tool_call.type}")
            return None
        return tool_call.function.name

    def get_function_args(self, tool_call: ToolCall) -> t.Dict[str, t.Any]:
        if not self.is_function() or not tool_call.function:
            return {}
        arguments = tool_call.function.arguments
        try:
            return json.loads(arguments)
        except json.JSONDecodeError as e:
            logger.debug("Failed to parse function arguments", exc_info=e)
            return {}

    def is_reached_limit(self) -> bool:
        return self.finish_reason == FINISH_REASON_LENGTH

    def to_dict(self) -> t.Dict[str, str]:
        return {
            "finish_reason": self.finish_reason,
            "message": self.message.model_dump_json(indent=2),
        }

    def get_message_str(self) -> str:
        return self.message.model_dump_json(indent=2)

    def __str__(self) -> str:
        result = (
            f"finish_reason: {self.finish_reason}\n"
            f"message: {self.get_message_str()}\n"
        )
        return result
