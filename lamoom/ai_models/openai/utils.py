import logging
import requests

import openai

from lamoom.ai_models.openai.exceptions import (
    OpenAIAuthenticationError,
    OpenAIBadRequestError,
    OpenAIChunkedEncodingError,
    OpenAIInternalError,
    OpenAIInvalidRequestError,
    OpenAiPermissionDeniedError,
    OpenAiRateLimitError,
    OpenAIResponseWasFilteredError,
    OpenAITimeoutError,
    OpenAIUnknownError,
    ConnectionCheckError,
)

logger = logging.getLogger(__name__)


def raise_openai_exception(
    exc: Exception,
) -> None:
    if isinstance(exc, requests.exceptions.ChunkedEncodingError):
        raise OpenAIChunkedEncodingError()

    if isinstance(exc, openai.APITimeoutError):
        raise OpenAITimeoutError()

    if isinstance(exc, openai.BadRequestError):
        if "response was filtered" in str(exc):
            raise OpenAIResponseWasFilteredError()
        if "Too many inputs" in str(exc):
            raise OpenAiRateLimitError()
        raise OpenAIInvalidRequestError()
    if isinstance(exc, openai.RateLimitError):
        raise OpenAiRateLimitError()

    if isinstance(exc, openai.AuthenticationError):
        raise OpenAIAuthenticationError()

    if isinstance(exc, openai.InternalServerError):
        raise OpenAIInternalError()

    if isinstance(exc, openai.PermissionDeniedError):
        raise OpenAiPermissionDeniedError()

    if isinstance(exc, openai.APIStatusError):
        raise OpenAIBadRequestError()

    if isinstance(exc, ConnectionError):
        raise ConnectionCheckError("websocket connection was lost")

    logger.error(
        "Unknown OPENAI error, please add it in raise_openai_rate_limit_exception",
        exc_info=exc,
    )
    raise OpenAIUnknownError()
