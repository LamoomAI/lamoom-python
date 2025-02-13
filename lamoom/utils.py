from decimal import Decimal
import json
import logging
import typing as t
from time import time

logger = logging.getLogger(__name__)


def parse_bool(value: any) -> bool:
    if type(value) == bool:
        return value
    return str(value).lower() in ("true", "1", "yes")


def current_timestamp_ms():
    return int(time() * 1000)


def resolve(prompt: str, context: t.Dict[str, str]) -> str:
    if not prompt or "{" not in prompt:
        return prompt
    # TODO: monitor how many values were not resolved and what values
    for key in context:
        prompt = prompt.replace(f"{{{key}}}", str(context[key]))
    return prompt


class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return str(o)
        return super(DecimalEncoder, self).default(o)
