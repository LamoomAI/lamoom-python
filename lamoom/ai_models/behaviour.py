import logging
from dataclasses import dataclass

from lamoom.ai_models.attempt_to_call import AttemptToCall

logger = logging.getLogger(__name__)


@dataclass
class AIModelsBehaviour:
    attempt: AttemptToCall
    fallback_attempts: list[AttemptToCall] = None


@dataclass
class PromptAttempts:
    ai_models_behaviour: AIModelsBehaviour
    current_attempt: AttemptToCall = None

    def initialize_attempt(self):
        if self.current_attempt is None:
            self.current_attempt = self.ai_models_behaviour.attempt
            self.fallback_index = 0  # Start fallback index at 0
            return self.current_attempt
        elif self.ai_models_behaviour.fallback_attempts:
            if self.fallback_index < len(self.ai_models_behaviour.fallback_attempts):
                self.current_attempt = self.ai_models_behaviour.fallback_attempts[self.fallback_index]
                self.fallback_index += 1
                return self.current_attempt
            else:
                self.current_attempt = None  # No more fallback attempts left
                return None

    def __str__(self) -> str:
        return f"Current attempt {self.current_attempt} from {len(self.ai_models_behaviour.attempts)}"
