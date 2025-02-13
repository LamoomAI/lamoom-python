import logging
import typing as t
import uuid
from dataclasses import dataclass

from lamoom.exceptions import ValueIsNotResolvedError
from lamoom.utils import resolve

logger = logging.getLogger(__name__)


@dataclass
class ValuesCost:
    values: t.List[str]
    cost: int


class ChatMessage:
    role: str
    content: str
    name: t.Optional[str] = None
    tool_calls: t.Dict[str, str]
    ref_name: t.Optional[str] = None
    ref_value: t.Optional[str] = None

    def is_not_empty(self):
        return bool(self.content or self.tool_calls)

    def is_empty(self):
        return not self.is_not_empty()

    def not_tool_calls(self):
        return not (self.tool_calls)

    def __init__(self, **kwargs):
        self.role = kwargs.get("role", "user")
        self.content = kwargs["content"]
        self.name = kwargs.get("name")
        self.tool_calls = kwargs.get("tool_calls") or {}

    def to_dict(self):
        result = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result


# can be multiple value
@dataclass(kw_only=True)
class ChatsEntity:
    content: str = ""
    role: str = "user"
    name: t.Optional[str] = None
    tool_calls: t.Dict[str, str] = None
    priority: int = 0
    required: bool = False
    is_multiple: bool = False
    while_fits: bool = False
    add_in_reverse_order: bool = False
    in_one_message: bool = False
    continue_if_doesnt_fit: bool = False
    add_if_fitted_labels: t.List[str] = None
    label: t.Optional[str] = None
    presentation: t.Optional[str] = None
    last_words: t.Optional[str] = None
    ref_name: t.Optional[str] = None
    ref_value: t.Optional[str] = None

    def __post_init__(self):
        self._uuid = uuid.uuid4().hex

    def resolve(self, context: t.Dict[str, t.Any]) -> t.List[ChatMessage]:
        result = []
        content = self.content
        if self.is_multiple:
            # should be just one value like {messages} in prompt
            prompt_value = content.strip().replace("{", "").replace("}", "").strip()
            values = context.get(prompt_value, [])
            if not values:
                return []
            if not isinstance(values, list):
                raise ValueIsNotResolvedError(
                    f"Invalid value {values } for prompt {content}. Should be multiple"
                )
            else:
                # verify that values are json list of ChatMessage
                try:
                    result = [
                        ChatMessage(**({"content": c} if isinstance(c, str) else c))
                        for c in values
                    ]
                except TypeError as e:
                    raise ValueIsNotResolvedError(
                        f"Invalid value { values } for prompt {content}. Error: {e}"
                    )
            return result

        content = resolve(content, context)
        if not content:
            return []
        return [
            ChatMessage(
                name=self.name,
                role=self.role,
                content=content,
                tool_calls=self.tool_calls,
                ref_name=self.ref_name,
                ref_value=self.ref_value,
            )
        ]

    def get_values(self, context: t.Dict[str, str]) -> t.List[ChatMessage]:
        try:
            values = self.resolve(context)
        except Exception as e:
            logger.error(
                f"Error resolving prompt {self.content}, error: {e}", exc_info=True
            )
            return []
        return values

    def dump(self):
        data = {
            "content": self.content,
            "role": self.role,
            "name": self.name,
            "tool_calls": self.tool_calls,
            "priority": self.priority,
            "required": self.required,
            "is_multiple": self.is_multiple,
            "while_fits": self.while_fits,
            "add_in_reverse_order": self.add_in_reverse_order,
            "in_one_message": self.in_one_message,
            "continue_if_doesnt_fit": self.continue_if_doesnt_fit,
            "add_if_fitted_labels": self.add_if_fitted_labels,
            "label": self.label,
            "presentation": self.presentation,
            "last_words": self.last_words,
            "ref_name": self.ref_name,
            "ref_value": self.ref_value,
        }
        for k, v in list(data.items()):
            if v is None:
                del data[k]
        return data

    @classmethod
    def load(cls, data):
        return cls(
            content=data.get("content"),
            role=data.get("role"),
            name=data.get("name"),
            tool_calls=data.get("tool_calls"),
            priority=data.get("priority"),
            required=data.get("required"),
            is_multiple=data.get("is_multiple"),
            while_fits=data.get("while_fits"),
            add_in_reverse_order=data.get("add_in_reverse_order"),
            in_one_message=data.get("in_one_message"),
            continue_if_doesnt_fit=data.get("continue_if_doesnt_fit"),
            add_if_fitted_labels=data.get("add_if_fitted_labels"),
            label=data.get("label"),
            presentation=data.get("presentation"),
            last_words=data.get("last_words"),
            ref_name=data.get("ref_name"),
            ref_value=data.get("ref_value"),
        )
