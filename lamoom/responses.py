from decimal import Decimal
import json
import logging
from dataclasses import dataclass, field
import typing as t

logger = logging.getLogger(__name__)


@dataclass
class Prompt(BasePrompt):
    id: str = None
    max_tokens: int = None
    min_sample_tokens: int = settings.DEFAULT_SAMPLE_MIN_BUDGET
    reserved_tokens_budget_for_sampling: int = None
    version: str = None
    # Add tool registry
    tool_registry: t.Dict[str, ToolDefinition] = field(default_factory=dict)

    def add_tool(self, tool: ToolDefinition):
        """Add a tool to this prompt's registry"""
        self.tool_registry[tool.name] = tool


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
        return self._response

    def get_message_str(self) -> str:
        return json.loads(self.response)
