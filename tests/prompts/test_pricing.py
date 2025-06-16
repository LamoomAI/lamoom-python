import json
import logging
import os
import time

from pytest import fixture
import dotenv
dotenv.load_dotenv(dotenv.find_dotenv())
from lamoom import Lamoom, Prompt
logger = logging.getLogger(__name__)


@fixture
def lamoom_client():

    lamoom = Lamoom()
    return lamoom

def stream_function(text, **kwargs):
    print(text)

def stream_check_connection(validate, **kwargs):
    return validate


def test_openai_pricing(lamoom_client: Lamoom):

    context = {
        'ideal_answer': "There are eight planets",
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }

    # initial version of the prompt
    prompt_id = f'unit-test_openai_pricing'
    lamoom_client.service.clear_cache()
    prompt = Prompt(id=prompt_id) 
    prompt.add("{text}", role='user')

    result_4o_mini = lamoom_client.call(prompt.id, context, "openai/o4-mini-mini", test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    
    assert result_4o_mini.metrics.prompt_tokens_used >= 0
    assert result_4o_mini.metrics.prompt_tokens_used >= 0
    

def test_claude_pricing(lamoom_client: Lamoom):

    context = {
        'ideal_answer': "There are eight planets",
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }

    # initial version of the prompt
    prompt_id = f'test'
    lamoom_client.service.clear_cache()
    prompt = Prompt(id=prompt_id) 
    prompt.add("{text}", role='user')
    
    result_haiku = lamoom_client.call(prompt.id, context, "claude/claude-3-5-haiku-latest", test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})

    assert result_haiku.metrics.prompt_tokens_used >= 0
    assert result_haiku.metrics.sample_tokens_used >= 0
