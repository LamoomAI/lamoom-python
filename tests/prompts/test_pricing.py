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

    result_4o = lamoom_client.call(prompt.id, context, "openai/gpt-4o", test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    result_4o_mini = lamoom_client.call(prompt.id, context, "openai/gpt-4o-mini", test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    
    assert result_4o.metrics.price_of_call > result_4o_mini.metrics.price_of_call
    

def test_claude_pricing(lamoom_client: Lamoom):

    context = {
        'ideal_answer': "There are eight planets",
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }

    # initial version of the prompt
    prompt_id = f'test-{time.time()}'
    lamoom_client.service.clear_cache()
    prompt = Prompt(id=prompt_id) 
    prompt.add("{text}", role='user')
    
    result_haiku = lamoom_client.call(prompt.id, context, "claude/claude-3-haiku-20240307", test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    result_sonnet = lamoom_client.call(prompt.id, context, "claude/claude-3-sonnet-20240229", test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    
    assert result_sonnet.metrics.price_of_call > result_haiku.metrics.price_of_call