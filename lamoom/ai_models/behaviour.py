import logging
import random
import typing as t
from copy import copy
from dataclasses import dataclass

from lamoom.ai_models.attempt_to_call import AttemptToCall
from lamoom.exceptions import BehaviourIsNotDefined

logger = logging.getLogger(__name__)


@dataclass
class AIModelsBehaviour:
    # if you have mutiple AI Models, you can distribute the load across them.
    # If you wish to use as a fallback attempt a model which is not in the list, you can use fallback_attempt
    attempts: list[AttemptToCall]
    fallback_attempt: AttemptToCall = None


@dataclass
class PromptAttempts:
    ai_models_behaviour: AIModelsBehaviour
    count_of_retries: t.Optional[int] = None
    count: int = 0
    current_attempt: AttemptToCall = None

    def __post_init__(self):
        if self.count_of_retries is None:
            self.count_of_retries = len(self.ai_models_behaviour.attempts) + int(
                bool(self.ai_models_behaviour.fallback_attempt)
            )

    def initialize_attempt(self, flag_increase_count: bool = True):
        if self.count > self.count_of_retries:
            raise BehaviourIsNotDefined(
                f"Count of retries {self.count_of_retries} exceeded {self.count}"
            )
        if (
            self.count == self.count_of_retries
            and self.ai_models_behaviour.fallback_attempt
        ):
            self.current_attempt = self.ai_models_behaviour.fallback_attempt
            return self.current_attempt
        sum_weight = sum(
            [attempt.weight for attempt in self.ai_models_behaviour.attempts]
        )
        random_weight = random.randint(0, sum_weight)
        for attempt in self.ai_models_behaviour.attempts:
            random_weight -= attempt.weight
            if random_weight <= 0:
                if flag_increase_count:
                    self.count += 1
                attempt.attempt_number = self.count
                self.current_attempt = copy(attempt)
                return self.current_attempt
        raise BehaviourIsNotDefined(
            f"Count of retries {self.count_of_retries} exceeded {self.count}"
        )

    def __str__(self) -> str:
        return f"Current attempt {self.current_attempt} from {len(self.ai_models_behaviour.attempts)}"
