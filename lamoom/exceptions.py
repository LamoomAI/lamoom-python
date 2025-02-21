class LamoomError(Exception):
    pass


class RetryableCustomError(LamoomError):
    pass


class LamoomPromptIsnotFoundError(LamoomError):
    pass


class BehaviourIsNotDefined(LamoomError):
    pass


class ConnectionLostError(LamoomError):
    pass


class ValueIsNotResolvedError(LamoomError):
    pass


class NotEnoughBudgetError(LamoomError):
    pass


class NotFoundPromptError(LamoomError):
    pass


class ProviderNotFoundError(LamoomError):
    pass


class NotParsedResponseException(LamoomError):
    pass

class APITokenNotProvided(LamoomError):
    pass
