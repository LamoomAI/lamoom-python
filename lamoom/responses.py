from decimal import Decimal
import json
import logging
from dataclasses import dataclass, field
import typing as t
from lamoom.ai_models.tools.base_tool import TOOL_CALL_NAME, TOOL_CALL_RESULT_NAME, ToolCallResult, ToolDefinition, format_tool_result_message

logger = logging.getLogger(__name__)

FINISH_REASON_LENGTH = "length"
FINISH_REASON_ERROR = "error"
FINISH_REASON_FINISH = "stop"
FINISH_REASON_TOOL_CALLS = "tool_calls"


@dataclass
class Prompt:
    messages: dict = None
    functions: dict = None
    max_tokens: int = 0
    temperature: Decimal = Decimal(0.0)
    top_p: Decimal = Decimal(0.0)


@dataclass
class Metrics:
    price_of_call: Decimal = None
    sample_tokens_used: int = None
    prompt_tokens_used: int = None
    ai_model_details: dict = None
    latency: int = None


@dataclass(kw_only=True)
class AIResponse:
    _response: str = ""
    original_result: object = None
    content: str = ""
    finish_reason: str = ""
    prompt: Prompt = field(default_factory=Prompt)
    metrics: Metrics = field(default_factory=Metrics)
    id: str = ""

    @property
    def response(self) -> str:
        return self._response or self.content or '{}'

    def get_message_str(self) -> str:
        return self.response


@dataclass(kw_only=True)
class StreamingResponse(AIResponse):
    is_detected_tool_call: bool = False
    last_detected_tool_call: t.Optional[dict] = None
    detected_tool_calls: list[t.Optional[dict]] = field(default_factory=list)
    tool_registry: t.Dict[str, ToolDefinition] = field(default_factory=dict)
    messages: t.List[dict] = field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def add_tool_result(self, tool_result: ToolCallResult):
        logger.info(f'TOOL_CALL: Added tool results {tool_result}')
        self.content +=f'\n<{TOOL_CALL_RESULT_NAME}="{tool_result.tool_name}">\n{json.dumps(tool_result.tool_name)}\n</{TOOL_CALL_RESULT_NAME}>\n'
        self.content += format_tool_result_message(tool_result)
        self.add_message("assistant", self.content)

    
    @property
    def response(self) -> str:
        return self.content

    @property
    def tool_calls(self) -> t.List[t.Optional[dict]]:
        return self.detected_tool_calls

    def get_function_name(self, tool_call: t.Optional[dict]) -> t.Optional[str]:
        if tool_call.type != "function":
            logger.error(f"function.type is not function: {tool_call.type}")
            return None
        return tool_call.function.name

    def get_function_args(self, tool_call: t.Optional[dict]) -> t.Dict[str, t.Any]:
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
            "message": json.dumps(self.messages, indent=2)
        }

    def get_message_str(self) -> str:
        return json.dumps(self.messages, indent=2)

    def __str__(self) -> str:
        result = (
            f"finish_reason: {self.finish_reason}\n"
            f"message: {self.get_message_str()}\n"
        )
        return result