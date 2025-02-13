import pytest
import requests
import openai
from unittest.mock import Mock
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
)
from lamoom.ai_models.openai.utils import raise_openai_exception

@pytest.fixture
def mock_response():
    return Mock()

def test_raise_openai_exception_with_chunked_encoding_error():
    with pytest.raises(OpenAIChunkedEncodingError):
        raise_openai_exception(requests.exceptions.ChunkedEncodingError())

def test_raise_openai_exception_with_timeout_error(mock_response: Mock):
    with pytest.raises(OpenAITimeoutError):
        raise_openai_exception(openai.APITimeoutError(request=mock_response))

def test_raise_openai_exception_with_bad_request_error_filtered_response(mock_response: Mock):
    with pytest.raises(OpenAIResponseWasFilteredError):
        raise_openai_exception(openai.BadRequestError(message="response was filtered", response=mock_response, body=None))

def test_raise_openai_exception_with_bad_request_error_rate_limit(mock_response: Mock):
    with pytest.raises(OpenAiRateLimitError):
        raise_openai_exception(openai.BadRequestError(message="Too many inputs", response=mock_response, body=None))

def test_raise_openai_exception_with_bad_request_error_invalid_request(mock_response: Mock):
    with pytest.raises(OpenAIInvalidRequestError):
        raise_openai_exception(openai.BadRequestError(message="Some other bad request error", response=mock_response, body=None))

def test_raise_openai_exception_with_rate_limit_error(mock_response: Mock):
    with pytest.raises(OpenAiRateLimitError):
        raise_openai_exception(openai.RateLimitError(response=mock_response, message="Rate limit error", body=None))

def test_raise_openai_exception_with_authentication_error(mock_response: Mock):
    with pytest.raises(OpenAIAuthenticationError):
        raise_openai_exception(openai.AuthenticationError(message="Authentication error", response=mock_response, body=None))

def test_raise_openai_exception_with_internal_server_error(mock_response: Mock):
    with pytest.raises(OpenAIInternalError):
        raise_openai_exception(openai.InternalServerError(message="Internal server error", response=mock_response, body=None))

def test_raise_openai_exception_with_permission_denied_error(mock_response: Mock):
    with pytest.raises(OpenAiPermissionDeniedError):
        raise_openai_exception(openai.PermissionDeniedError(message="Permission denied error", response=mock_response, body=None))

def test_raise_openai_exception_with_api_status_error(mock_response: Mock):
    with pytest.raises(OpenAIBadRequestError):
        raise_openai_exception(openai.APIStatusError(message="API status error", response=mock_response, body=None))

def test_raise_openai_exception_with_unknown_error():
    with pytest.raises(OpenAIUnknownError):
        class UnknownError(Exception):
            pass
        raise_openai_exception(UnknownError())
