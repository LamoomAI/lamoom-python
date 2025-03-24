from decimal import Decimal
import json
import logging
from dataclasses import dataclass, field
from functools import cached_property
import typing as t
from lamoom.exceptions import LamoomError
logger = logging.getLogger(__name__)


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

@dataclass
class Tag:
    start_tag: str
    end_tag: str
    include_tag: bool
    is_right_find_end_ind: bool = False


@dataclass(kw_only=True)
class AIResponse:
    _response: str = ""
    original_result: object = None
    content: str = ""
    finish_reason: str = ""
    prompt: Prompt = field(default_factory=Prompt)
    metrics: Metrics = field(default_factory=Metrics)
    id: str = ""
    errors: t.Optional[t.List[LamoomError]] = None
    attemps: t.Optional[t.List] = None

    @property
    def response(self) -> str:
        return self._response

    def get_message_str(self) -> str:
        return json.loads(self.response)

    @cached_property
    def json_list(self) -> t.List[t.Dict]:

        tags = [Tag("```json", "\n```", 0), Tag("```json", "```", 0)]  
        return self._get_tagged_content_list(tags)

    @cached_property
    def xml_list(self) -> t.List[t.Dict]:

        tags = [Tag("```xml", "\n```", 0), Tag("```xml", "```", 0)]
        return self._get_tagged_content_list(tags)

    @cached_property
    def yaml_list(self) -> t.List[t.Dict]:

        tags = [Tag("```yaml", "\n```", 0), Tag("```yaml", "```", 0)]
        return self._get_tagged_content_list(tags)

    def _get_tagged_content_list(self, tags: Tag) -> t.List[t.Dict]:

        content_list = []
        start_from = 0
        while True:
            response_tagged, start_from, end_ind = self._get_format_from_response(tags, start_from)
            if response_tagged:
                content_list.append({
                        "content": response_tagged,
                        "start_ind": start_from,
                        "end_ind": end_ind
                    })
                start_from = end_ind
            else:
                break
        return content_list

    def _get_format_from_response(self, tags: list, start_from: int = 0):

        start_ind, end_ind = 0, -1
        content = self.response[start_from:]
        for t in tags:
            start_ind = content.find(t.start_tag)
            if t.is_right_find_end_ind:
                end_ind = content.rfind(t.end_tag, start_ind + len(t.start_tag))
            else:
                end_ind = content.find(t.end_tag, start_ind + len(t.start_tag))
            if start_ind != -1:
                try:
                    if t.include_tag:
                        end_ind += len(t.end_tag)
                    else:
                        start_ind += len(t.start_tag)
                    response_tagged = content[start_ind:end_ind].strip()
                    return response_tagged, start_from + start_ind, start_from + end_ind
                except Exception as e:
                    logger.exception(f"Couldn't parse json:\n{content}")
        return None, 0, -1
