from dataclasses import dataclass
import json
import logging

import yaml

from lamoom.exceptions import NotParsedResponseException
from lamoom.responses import AIResponse

logger = logging.getLogger(__name__)


@dataclass
class Tag:
    start_tag: str
    end_tag: str
    include_tag: bool
    is_right_find_end_ind: bool = False


@dataclass
class TaggedContent:
    content: str
    start_ind: int
    end_ind: int
    parsed_content: any = None


def get_yaml_from_response(response: AIResponse):
    content, start_ind, end_ind = _get_format_from_response(
        response, [Tag("```yaml", "```", 0, 0), Tag("```", "```", 0, 0)]
    )
    parsed_content = None
    if content:
        try:
            parsed_content = yaml.safe_load(content)
        except Exception as e:
            logger.exception(f"Couldn't parse yaml:\n{content}")
        return TaggedContent(
            content=content,
            parsed_content=parsed_content,
            start_ind=start_ind,
            end_ind=end_ind,
        )


def get_json_from_response(response: AIResponse, start_from: int = 0) -> TaggedContent:
    content, start_ind, end_ind = _get_format_from_response(
        response,
        [Tag("```json", "\n```", 0), Tag("```json", "```", 0), Tag("{", "}", 1)],
        start_from=start_from,
    )
    if content:
        try:
            json_response = eval(content)
            return TaggedContent(
                content=content,
                parsed_content=json_response,
                start_ind=start_ind,
                end_ind=end_ind,
            )
        except Exception as e:
            try:
                json_response = json.loads(content)
                return TaggedContent(
                    content=content,
                    parsed_content=json_response,
                    start_ind=start_ind,
                    end_ind=end_ind,
                )
            except Exception as e:
                logger.exception(f"Couldn't parse json:\n{content}")
                raise NotParsedResponseException()


def _get_format_from_response(
    response: AIResponse, tags: list[Tag], start_from: int = 0
):
    start_ind, end_ind = 0, -1
    content = response.response[start_from:]
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
