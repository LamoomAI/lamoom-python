from datetime import datetime
import logging
import typing as t
from dataclasses import dataclass, field
import requests
from lamoom.ai_models.tools.base_tool import inject_tool_prompts
from lamoom.settings import LAMOOM_API_URI
from lamoom import Secrets, settings    
from lamoom.ai_models.ai_model import AI_MODELS_PROVIDER
from lamoom.ai_models.attempt_to_call import AttemptToCall
from lamoom.ai_models.behaviour import AIModelsBehaviour, PromptAttempts
from lamoom.ai_models.openai.azure_models import AzureAIModel
from lamoom.ai_models.claude.claude_model import ClaudeAIModel
from lamoom.ai_models.openai.openai_models import OpenAIModel

from lamoom.exceptions import (
    LamoomPromptIsnotFoundError,
    RetryableCustomError,
    StopStreamingError
)
from lamoom.services.SaveWorker import SaveWorker
from lamoom.prompt.prompt import Prompt

from lamoom.responses import AIResponse
from lamoom.services.lamoom import LamoomService
import json

logger = logging.getLogger(__name__)

BASE_URL_MAPPING = {
    'gemini': "https://generativelanguage.googleapis.com/v1beta/openai/"
}

@dataclass
class Lamoom:
    api_token: str = None
    openai_key: str = None
    openai_org: str = None
    claude_key: str = None
    gemini_key: str = None
    azure_keys: t.Dict[str, str] = None
    custom_keys: t.Dict[str, str] = field(default_factory=dict)
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
        if not self.custom_keys:
            if self.secrets.custom_keys:
                logger.debug(f"Using Custom keys from secrets")
                self.custom_keys = self.secrets.custom_keys
            else:
                logger.debug(f"Custom keys not found in secrets")
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
            self.clients[AI_MODELS_PROVIDER.OPENAI.value] = {
                "organization": self.openai_org,
                "api_key": self.openai_key,
            }
        if self.azure_keys:
            if not self.clients.get(AI_MODELS_PROVIDER.AZURE.value):
                self.clients[AI_MODELS_PROVIDER.AZURE.value] = {}
            for realm, key_data in self.azure_keys.items():
                self.clients[AI_MODELS_PROVIDER.AZURE.value][realm] = {
                    "api_version": key_data.get("api_version", "2023-07-01-preview"),
                    "azure_endpoint": key_data["url"],
                    "api_key": key_data["key"],
                }
                logger.debug(f"Initialized Azure client for {realm} {key_data['url']}")
        if self.claude_key:
            self.clients[AI_MODELS_PROVIDER.CLAUDE.value] = {"api_key": self.claude_key}
        if self.gemini_key:
            self.clients[AI_MODELS_PROVIDER.GEMINI.value] = {
                "api_key": self.gemini_key,
                "base_url": BASE_URL_MAPPING.get(AI_MODELS_PROVIDER.GEMINI.value)
            }
        # Initialize custom providers from environment
        for provider_name, provider_config in self.custom_keys.items():
            logger.info(f"Initializing custom provider {provider_name} {provider_config.get('base_url')}")
            provider_key = f"custom_{provider_name}"
            if provider_key not in self.clients:
                self.clients[provider_key] = {
                    "api_key": provider_config.get("key"),
                    "base_url": provider_config.get("base_url")
                }
                logger.debug(f"Initialized custom provider {provider_key} {provider_config.get('base_url')}")
        self.worker = SaveWorker()

    def create_test(
        self, prompt_id: str, context: t.Dict[str, str], ideal_answer: str = None, model_name: str = None
    ):
        """
        Create new test
        """

        url = f"{LAMOOM_API_URI}/lib/tests?createTest"
        headers = {"Authorization": f"Token {self.api_token}"}
        if "ideal_answer" in context:
            ideal_answer = context["ideal_answer"]

        data = {
            "prompt_id": prompt_id,
            "ideal_answer": ideal_answer,
            "model_name": model_name,
            "test_context": context,
        }
        json_data = json.dumps(data)
        response = requests.post(url, headers=headers, data=json_data)

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(response)
            
    def extract_provider_name(self, model: str) -> dict:
        parts = model.split("/")

        if "openai" in parts[0].lower() and len(parts) == 2:
            return {
                'provider':  parts[0].lower(),
                'model_name': parts[1]
            }
    
        elif "azure" in parts[0].lower() and len(parts) == 3:
            model_provider, realm, model_name = parts
            return {
                'provider': model_provider.lower(),
                'model_name': model_name,
                'realm': realm,
            }
        elif "claude" in parts[0].lower() and len(parts) == 2:
            return {
                'provider': 'claude',
                'model_name': parts[1]
            }
        elif "gemini" in parts[0].lower() and len(parts) == 2:
            return {
                'provider': 'gemini',
                'model_name': parts[1]
            }
        elif "custom" in parts[0].lower() and len(parts) >= 3:
                model_name = '/'.join(parts[2:])
                provider_name = parts[1].lower()
                # Check if this is a registered custom provider
                if provider_name in self.custom_keys:
                    return {
                        'provider': f"custom_{provider_name}",
                        'model_name': model_name,
                        'realm': None,
                    }
        raise Exception(f"Unknown model: {model}")
    
    def get_default_context(self):
        return {
            'current_datetime_strftime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'timezone': datetime.now().astimezone().tzname()
        }
    
    def get_context(self, context: dict):
        return {
            **self.get_default_context(),
            **context
        }

    def init_attempt(self, model_info: dict) -> AttemptToCall:
        provider = model_info['provider']
        model_name = model_info['model_name']
                
        if provider == AI_MODELS_PROVIDER.CLAUDE.value:
            return AttemptToCall(
                    ai_model=ClaudeAIModel(
                        model=model_name,
                    ),
                    weight=100,
                )
        elif provider == AI_MODELS_PROVIDER.OPENAI.value:
            return AttemptToCall(
                    ai_model=OpenAIModel(
                        model=model_name
                    ),
                    weight=100,
                )
        elif provider == AI_MODELS_PROVIDER.GEMINI.value:
            return AttemptToCall(
                    ai_model=OpenAIModel(
                        model=model_name,
                        provider=AI_MODELS_PROVIDER.GEMINI,
                    ),
                    weight=100,
                )
        elif provider.startswith('custom_'):
            # Handle custom provider format
            return AttemptToCall(
                    ai_model=OpenAIModel(
                        model=model_name,
                        provider=AI_MODELS_PROVIDER.CUSTOM,
                        _provider_name=model_info['provider']
                    ),
                    weight=100,
                )
        elif provider == AI_MODELS_PROVIDER.AZURE.value:
            return AttemptToCall(
                    ai_model=AzureAIModel(
                        realm=model_info['realm'],
                        deployment_id=model_name,
                    ),
                    weight=100,
                )
    
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
        stream_params: dict = {}
    ) -> AIResponse:
        """
        Call flow prompt with context and behaviour
        """

        logger.debug(f"Calling {prompt_id}")
        prompt = self.get_prompt(prompt_id, version)
        
        behaviour = self.init_behavior(model)
        
        logger.info(behaviour)
        
        prompt_attempts = PromptAttempts(behaviour)

        while prompt_attempts.initialize_attempt():
            current_attempt = prompt_attempts.current_attempt
            user_prompt = prompt.create_prompt(current_attempt)
            calling_context = self.get_context(context)
            # Inject tool prompts into first message
            calling_messages = user_prompt.resolve(calling_context, prompt.tool_registry)
            messages = calling_messages.get_messages()
            messages = inject_tool_prompts(messages, list(prompt.tool_registry.values()), calling_context)
            logger.info(f'self.clients: {self.clients}, [current_attempt.ai_model.provider_name]: {current_attempt.ai_model.provider_name}')
            for _ in range(0, count_of_retries):
                try:
                    result = current_attempt.ai_model.call(
                        messages,
                        calling_messages.max_sample_budget,
                        tool_registry=prompt.tool_registry,
                        stream_function=stream_function,
                        check_connection=check_connection,
                        stream_params=stream_params,
                        client_secrets=self.clients[current_attempt.ai_model.provider_name],
                        modelname=model,
                        prompt=prompt,
                        context=context,
                        test_data=test_data,
                        client=self,
                        **params,
                    )
                    return result
                except RetryableCustomError as e:
                    logger.exception(
                        f"Attempt failed: {prompt_attempts.current_attempt} with retryable error: {e}"
                    )
                    break
                except StopStreamingError as e:
                    logger.exception(
                        f"Attempt Stopped: {prompt_attempts.current_attempt} with non-retryable error: {e}"
                    )
                    raise e
                except Exception as e:
                    logger.exception(
                        f"Attempt failed: {prompt_attempts.current_attempt} with non-retryable error: {e}"
                    )
                    raise e
                    
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
