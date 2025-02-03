import logging
import typing as t
from dataclasses import dataclass
from decimal import Decimal
import requests
import time
from flow_prompt.settings import FLOW_PROMPT_API_URI
from flow_prompt import Secrets, settings
from flow_prompt.ai_models.ai_model import AI_MODELS_PROVIDER
from flow_prompt.ai_models.attempt_to_call import AttemptToCall
from flow_prompt.ai_models.behaviour import AIModelsBehaviour, PromptAttempts
from flow_prompt.exceptions import (
    FlowPromptIsnotFoundError,
    RetryableCustomError
)
from flow_prompt.services.SaveWorker import SaveWorker
from flow_prompt.prompt.pipe_prompt import PipePrompt
from flow_prompt.prompt.user_prompt import UserPrompt
from flow_prompt.responses import AIResponse
from flow_prompt.services.flow_prompt import FlowPromptService
from flow_prompt.utils import current_timestamp_ms
import json

logger = logging.getLogger(__name__)

@dataclass
class FlowPrompt:
    api_token: str = None
    openai_key: str = None
    openai_org: str = None
    claude_key: str = None
    gemini_key: str = None
    azure_keys: t.Dict[str, str] = None
    nebius_key: str = None
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
        self.service = FlowPromptService()
        if self.nebius_key:
            self.clients[AI_MODELS_PROVIDER.NEBIUS] = {
                'api_key': self.nebius_key,
                'base_url': "https://api.studio.nebius.ai/v1/"
            }
        if self.openai_key:
            self.clients[AI_MODELS_PROVIDER.OPENAI] = {
                'organization': self.openai_org,
                'api_key': self.openai_key,
            }
        if self.azure_keys:
            if not self.clients.get(AI_MODELS_PROVIDER.AZURE):
                self.clients[AI_MODELS_PROVIDER.AZURE] = {}
            for realm, key_data in self.azure_keys.items():
                self.clients[AI_MODELS_PROVIDER.AZURE][realm] = {
                    'api_version': key_data.get("api_version", "2023-07-01-preview"),
                    'azure_endpoint': key_data["url"],
                    'api_key': key_data["key"],
                }
                logger.debug(f"Initialized Azure client for {realm} {key_data['url']}")
        if self.claude_key:
            self.clients[AI_MODELS_PROVIDER.CLAUDE] = {'api_key': self.claude_key}
        if self.gemini_key:
            self.clients[AI_MODELS_PROVIDER.GEMINI] = {'api_key': self.gemini_key}
        self.worker = SaveWorker()

    
    def create_test(self, 
        prompt_id: str,
        test_context: t.Dict[str, str],
        ideal_answer: str = None
    ):
        '''
        Create new test
        '''
        
        url = f'{FLOW_PROMPT_API_URI}lib/tests?createTest'
        headers = {"Authorization": f"Token {self.api_token}"}
        if 'ideal_answer' in test_context:
            ideal_answer = test_context['ideal_answer']
            
        data = {
            'prompt_id': prompt_id,
            'ideal_answer': ideal_answer,
            'test_context': test_context
        }
        json_data = json.dumps(data)
        response = requests.post(url, headers=headers, data=json_data)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(response)


    def call(
        self,
        prompt_id: str,
        context: t.Dict[str, str],
        behaviour: AIModelsBehaviour,
        params: t.Dict[str, t.Any] = {},
        version: str = None,
        count_of_retries: int = None,
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
        pipe_prompt = self.get_pipe_prompt(prompt_id, version)
        prompt_attempts = PromptAttempts(behaviour, count_of_retries=count_of_retries)

        while prompt_attempts.initialize_attempt():
            current_attempt = prompt_attempts.current_attempt
            user_prompt = pipe_prompt.create_prompt(current_attempt)
            calling_messages = user_prompt.resolve(context)
            
            """
            Create CI/CD when calling first time
            """
        
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
                result.metrics.price_of_call = self.get_price(
                    current_attempt,
                    sample_budget,
                    calling_messages.prompt_budget,
                )
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
                        pipe_prompt.service_dump(),
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
                logger.exception(
                    f"Attempt failed: {prompt_attempts.current_attempt} with non-retryable error: {e}"
                )
                raise e

    def add_ideal_answer(
        self,
        response_id: str,
        ideal_answer: str
    ):
        response = FlowPromptService.update_response_ideal_answer(
            self.api_token, response_id, ideal_answer
        )
        
        return response
    
    def update_overview(self, overview: str, user_id: str = None):
        """Update user's overview

        Args:
            user_id (str): user id,
            overview (str): new overview to replace the old one
        """
        
        response = FlowPromptService.update_user_overview(user_id, overview, self.api_token)
        
        return response
    
    def get_file_names(self, prefix: str, user_id: str = None):
        """Fetch all filenames of the given user

        Args:
            prefix (str): s3 bucket folder name to fetch from
            user_id (str): user identifier

        Returns:
            list: list of file names
        """
        response = FlowPromptService.get_file_names(prefix, user_id, self.api_token)
        
        return response

    def get_files(self, paths: list[str], user_id: str = None):
        """Method to fetch file contents by the provided s3 paths

        Args:
            paths (list[str]): paths to s3 bucket files
            user_id (str): user identifier
        
        Returns: 
            dict: key = path, value: file content 
        """
        
        response = FlowPromptService.get_files(paths, user_id, self.api_token)
        
        return response
    
    def save_files(self, files: dict, user_id: str = None):
        """Method to save files into FPS S3 bucket

        Args:
            files (dict): dictionary where key = file_name (relative path), val = file_content
        """
        
        response = FlowPromptService.save_files(files, user_id, self.api_token)
        
        return response
    
    
    def get_pipe_prompt(self, prompt_id: str, version: str = None) -> PipePrompt:
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
                return PipePrompt.service_load(response.prompt)
            except Exception as e:
                logger.exception(f"Error while getting prompt {prompt_id}: {e}")
                if prompt:
                    return prompt
                else:
                    logger.exception(f"Prompt {prompt_id} not found")
                    raise FlowPromptIsnotFoundError()

        else:
            return settings.PIPE_PROMPTS[prompt_id]

    def calculate_budget_for_text(self, user_prompt: UserPrompt, text: str) -> int:
        if not text:
            return 0
        return len(user_prompt.encoding.encode(text))

    def get_price(
        self, attempt: AttemptToCall, sample_budget: int, prompt_budget: int
    ) -> Decimal:
        return attempt.ai_model.get_prompt_price(prompt_budget) + attempt.ai_model.get_sample_price(prompt_budget, sample_budget)