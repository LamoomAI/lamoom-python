import logging
import typing as t
from dataclasses import dataclass
from decimal import Decimal
import requests
import time
from lamoom.settings import LAMOOM_API_URI
from lamoom import Secrets, settings    
from lamoom.ai_models.ai_model import AI_MODELS_PROVIDER
from lamoom.ai_models.attempt_to_call import AttemptToCall
from lamoom.ai_models.behaviour import AIModelsBehaviour, PromptAttempts
from lamoom.ai_models.openai.azure_models import AzureAIModel
from lamoom.ai_models.claude.claude_model import ClaudeAIModel
from lamoom.ai_models.openai.openai_models import OpenAIModel
from lamoom.ai_models.constants import C_16K

from lamoom.exceptions import (
    LamoomPromptIsnotFoundError,
    RetryableCustomError
)
from lamoom.services.SaveWorker import SaveWorker
from lamoom.prompt.prompt import Prompt
from lamoom.prompt.user_prompt import UserPrompt

from lamoom.responses import AIResponse
from lamoom.services.lamoom import LamoomService
from lamoom.utils import current_timestamp_ms
import json

logger = logging.getLogger(__name__)


@dataclass
class Lamoom:
    api_token: str = None
    openai_key: str = None
    openai_org: str = None
    claude_key: str = None
    gemini_key: str = None
    azure_keys: t.Dict[str, str] = None
    secrets: Secrets = None

    clients = {}

    def __post_init__(self):
        self.secrets = Secrets()
        if not self.azure_keys:
            if self.secrets.azure_keys:
                logger.debug(f"Using Azure keys from secrets")
                self.azure_keys = self.secrets.azure_keys
            else:
                logger.debug(f"Azure keys not found in secrets")
        if not self.api_token and self.secrets.API_TOKEN:
            logger.debug(f"Using API token from secrets")
            self.api_token = self.secrets.API_TOKEN
        if not self.openai_key and self.secrets.OPENAI_API_KEY:
            logger.debug(f"Using OpenAI API key from secrets")
            self.openai_key = self.secrets.OPENAI_API_KEY
        if not self.openai_org and self.secrets.OPENAI_ORG:
            logger.debug(f"Using OpenAI organization from secrets")
            self.openai_org = self.secrets.OPENAI_ORG
        if not self.gemini_key and self.secrets.GEMINI_API_KEY:
            logger.debug(f"Using Gemini API key from secrets")
            self.gemini_key = self.secrets.GEMINI_API_KEY
        if not self.claude_key and self.secrets.CLAUDE_API_KEY:
            logger.debug(f"Using Claude API key from secrets")
            self.claude_key = self.secrets.CLAUDE_API_KEY
        self.service = LamoomService()
        if self.openai_key:
            self.clients[AI_MODELS_PROVIDER.OPENAI] = {
                "organization": self.openai_org,
                "api_key": self.openai_key,
            }
        if self.azure_keys:
            if not self.clients.get(AI_MODELS_PROVIDER.AZURE):
                self.clients[AI_MODELS_PROVIDER.AZURE] = {}
            for realm, key_data in self.azure_keys.items():
                self.clients[AI_MODELS_PROVIDER.AZURE][realm] = {
                    "api_version": key_data.get("api_version", "2023-07-01-preview"),
                    "azure_endpoint": key_data["url"],
                    "api_key": key_data["key"],
                }
                logger.debug(f"Initialized Azure client for {realm} {key_data['url']}")
        if self.claude_key:
            self.clients[AI_MODELS_PROVIDER.CLAUDE] = {"api_key": self.claude_key}
        if self.gemini_key:
            self.clients[AI_MODELS_PROVIDER.GEMINI] = {"api_key": self.gemini_key}
        self.worker = SaveWorker()

    def create_test(
        self, prompt_id: str, test_context: t.Dict[str, str], ideal_answer: str = None
    ):
        """
        Create new test
        """

        url = f"{LAMOOM_API_URI}/lib/tests?createTest"
        headers = {"Authorization": f"Token {self.api_token}"}
        if "ideal_answer" in test_context:
            ideal_answer = test_context["ideal_answer"]

        data = {
            "prompt_id": prompt_id,
            "ideal_answer": ideal_answer,
            "test_context": test_context,
        }
        json_data = json.dumps(data)
        response = requests.post(url, headers=headers, data=json_data)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(response)
            
    def init_attempt(self, model_info: dict) -> AttemptToCall:
        provider = model_info['provider']
        model_name = model_info['model_name']
                
        if provider in [AI_MODELS_PROVIDER.OPENAI.value, 
                        AI_MODELS_PROVIDER.GEMINI.value, 
                        AI_MODELS_PROVIDER.NEBIUS.value]:
            return AttemptToCall(
                    ai_model=OpenAIModel(
                        provider=AI_MODELS_PROVIDER(provider),
                        model=model_name,
                    ),
                    weight=100,
                )   
        elif provider == AI_MODELS_PROVIDER.CLAUDE.value:
            return AttemptToCall(
                    ai_model=ClaudeAIModel(
                        model=model_name,
                    ),
                    weight=100,
                )
        else:
            return AttemptToCall(
                    ai_model=AzureAIModel(
                        realm=model_info['realm'],
                        deployment_id=model_name,
                    ),
                    weight=100,
                )
    
    def extract_provider_name(self, model: str) -> dict:
        parts = model.split("/")
    
        if "azure" in parts[0].lower() and len(parts) == 3:
            model_provider, realm, model_name = parts
        else:
            model_provider, model_name = parts
            realm = None
        
        return {
            'provider': model_provider.lower(),
            'model_name': model_name,
            'realm': realm
        }

    def init_behavior(self, model: str) -> AIModelsBehaviour:
        main_model_info = self.extract_provider_name(model)
        
        main_attempt = self.init_attempt(main_model_info)
        
        fallback_attempts = []
        for model in settings.FALLBACK_MODELS:
            model_info = self.extract_provider_name(model)
            fallback_attempts.append(self.init_attempt(model_info))
        
        return AIModelsBehaviour(
            attempt=main_attempt,
            fallback_attempts=fallback_attempts
        )
        
        
    def call(
        self,
        prompt_id: str,
        context: t.Dict[str, str],
        model: str,
        params: t.Dict[str, t.Any] = {},
        version: str = None,
        count_of_retries: int = 5,
        test_data: dict = {},
        stream_function: t.Callable = None,
        check_connection: t.Callable = None,
        stream_params: dict = {},
    ) -> AIResponse:
        """
        Call flow prompt with context and behaviour
        """

        logger.debug(f"Calling {prompt_id}")
        start_time = current_timestamp_ms()
        prompt = self.get_prompt(prompt_id, version)
        
        behaviour = self.init_behavior(model)
        
        logger.info(behaviour)
        
        prompt_attempts = PromptAttempts(behaviour)

        while prompt_attempts.initialize_attempt():
            current_attempt = prompt_attempts.current_attempt
            user_prompt = prompt.create_prompt(current_attempt)
            calling_messages = user_prompt.resolve(context)
            
            for _ in range(0, count_of_retries):
                try:
                    result = current_attempt.ai_model.call(
                        calling_messages.get_messages(),
                        calling_messages.max_sample_budget,
                        stream_function=stream_function,
                        check_connection=check_connection,
                        stream_params=stream_params,
                        client_secrets=self.clients[current_attempt.ai_model.provider],
                        **params,
                    )

                    sample_budget = self.calculate_budget_for_text(
                        user_prompt, result.get_message_str()
                    )
                    
                    try:
                        result.metrics.price_of_call = self.get_price(
                            current_attempt,
                            sample_budget,
                            calling_messages.prompt_budget,
                        )
                    except Exception as e:
                        logger.exception(f"Error while getting price: {e}")
                        result.metrics.price_of_call = 0
                    result.metrics.sample_tokens_used = sample_budget
                    result.metrics.prompt_tokens_used = calling_messages.prompt_budget
                    result.metrics.ai_model_details = (
                        current_attempt.ai_model.get_metrics_data()
                    )
                    result.metrics.latency = current_timestamp_ms() - start_time

                    if settings.USE_API_SERVICE and self.api_token:
                        timestamp = int(time.time() * 1000)
                        result.id = f"{prompt_id}#{timestamp}"
                        
                        self.worker.add_task(
                            self.api_token,
                            prompt.service_dump(),
                            context,
                            result,
                            test_data
                        )
                    return result
                except RetryableCustomError as e:
                    logger.error(
                        f"Attempt failed: {prompt_attempts.current_attempt} with retryable error: {e}"
                    )
                except Exception as e:
                    logger.error(
                        f"Attempt failed: {prompt_attempts.current_attempt} with non-retryable error: {e}"
                    )
                    
        logger.exception(
            "Prompt call failed, no attempts worked"
        )
        raise Exception

    def get_prompt(self, prompt_id: str, version: str = None) -> Prompt:
        """
        if the user has keys:  lib -> service: get_actual_prompt(local_prompt) -> Service:
        generates hash of the prompt;
        check in Redis if that record is the latest; if yes -> return 200, else
        checks if that record exists with that hash;
        if record exists and it's not the last - then we load the latest published prompt; - > return  200 + the last record
        add a new record in storage, and adding that it's the latest published prompt; -> return 200
        update redis with latest record;
        """
        logger.debug(f"Getting pipe prompt {prompt_id}")
        if (
            settings.USE_API_SERVICE
            and self.api_token
            and settings.RECEIVE_PROMPT_FROM_SERVER
        ):
            prompt_data = None
            prompt = settings.PIPE_PROMPTS.get(prompt_id)
            if prompt:
                prompt_data = prompt.service_dump()
            try:
                response = self.service.get_actual_prompt(
                    self.api_token, prompt_id, prompt_data, version
                )
                if not response.is_taken_globally:
                    prompt.version = response.version
                    return prompt
                response.prompt["version"] = response.version
                return Prompt.service_load(response.prompt)
            except Exception as e:
                logger.exception(f"Error while getting prompt {prompt_id}: {e}")
                if prompt:
                    return prompt
                else:
                    logger.exception(f"Prompt {prompt_id} not found")
                    raise LamoomPromptIsnotFoundError()

        else:
            return settings.PIPE_PROMPTS[prompt_id]


    def add_ideal_answer(
        self,
        response_id: str,
        ideal_answer: str
    ):
        response = LamoomService.update_response_ideal_answer(
            self.api_token, response_id, ideal_answer
        )
        
        return response
    
    def calculate_budget_for_text(self, user_prompt: UserPrompt, text: str) -> int:
        if not text:
            return 0
        return len(user_prompt.encoding.encode(text))

    def get_price(
        self, attempt: AttemptToCall, sample_budget: int, prompt_budget: int
    ) -> Decimal:
        data = {
                "provider": attempt.ai_model.provider.value,
                "model": attempt.ai_model.name,
                "output_tokens": sample_budget,
                "input_tokens": prompt_budget,
        }
        
        response = requests.post(
            f"{LAMOOM_API_URI}/lib/pricing",
            data=json.dumps(data),
        )
        
        if response.status_code != 200:
            return 0
        
        return response.json()["price"]