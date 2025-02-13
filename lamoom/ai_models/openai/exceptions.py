from lamoom.exceptions import LamoomError, RetryableCustomError


class OpenAIChunkedEncodingError(RetryableCustomError):
    pass


class OpenAITimeoutError(RetryableCustomError):
    pass


class OpenAIResponseWasFilteredError(RetryableCustomError):
    pass


class OpenAIAuthenticationError(RetryableCustomError):
    pass


class OpenAIInternalError(RetryableCustomError):
    pass


class OpenAiRateLimitError(RetryableCustomError):
    pass


class OpenAiPermissionDeniedError(RetryableCustomError):
    pass


class OpenAIUnknownError(RetryableCustomError):
    pass


### Non-retryable Errors ###
class OpenAIInvalidRequestError(LamoomError):
    pass


class OpenAIBadRequestError(LamoomError):
    pass


class ConnectionCheckError(LamoomError):
    pass
