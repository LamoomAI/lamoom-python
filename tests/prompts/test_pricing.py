import json
import logging
import os
import time

from pytest import fixture
import dotenv
dotenv.load_dotenv(dotenv.find_dotenv())
from lamoom import Lamoom, behaviour, Prompt, AttemptToCall, AzureAIModel, ClaudeAIModel, C_128K
logger = logging.getLogger(__name__)


@fixture
def lamoom_client():

    lamoom = Lamoom()
    return lamoom


@fixture
def openai_behaviour_4o():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=AzureAIModel(
                    realm='useast',
                    deployment_id="gpt-4o",
                    max_tokens=C_128K,
                    support_functions=True,
                ),
                weight=100,
            )
        ]
    )
    
@fixture
def openai_behaviour_4o_mini():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=AzureAIModel(
                    realm='useast',
                    deployment_id="gpt-4o-mini",
                    max_tokens=C_128K,
                    support_functions=True,
                ),
                weight=100,
            )
        ]
    )

@fixture
def claude_behaviour_haiku():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=ClaudeAIModel(
                    model="claude-3-haiku-20240307",
                    max_tokens=4096                
                ),
                weight=100,
            ),
        ]
    )
    
@fixture
def claude_behaviour_sonnet():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=ClaudeAIModel(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4096                
                ),
                weight=100,
            ),
        ]
    )

def stream_function(text, **kwargs):
    print(text)

def stream_check_connection(validate, **kwargs):
    return validate


def test_openai_pricing(lamoom_client: Lamoom, openai_behaviour_4o: behaviour.AIModelsBehaviour, openai_behaviour_4o_mini: behaviour.AIModelsBehaviour):

    context = {
        'ideal_answer': "There are eight planets",
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }

    # initial version of the prompt
    prompt_id = f'unit-test_openai_pricing'
    lamoom_client.service.clear_cache()
    prompt = Prompt(id=prompt_id) 
    prompt.add("{text}", role='user')

    result_4o = lamoom_client.call(prompt.id, context, openai_behaviour_4o, test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    result_4o_mini = lamoom_client.call(prompt.id, context, openai_behaviour_4o_mini, test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    
    assert result_4o.metrics.price_of_call > result_4o_mini.metrics.price_of_call
    

def test_claude_pricing(lamoom_client, claude_behaviour_haiku, claude_behaviour_sonnet):

    context = {
        'ideal_answer': "There are eight planets",
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }

    # initial version of the prompt
    prompt_id = f'test-{time.time()}'
    lamoom_client.service.clear_cache()
    prompt = Prompt(id=prompt_id) 
    prompt.add("{text}", role='user')
    
    lamoom_client.call(prompt.id, context, claude_behaviour_haiku, test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    lamoom_client.call(prompt.id, context, claude_behaviour_sonnet, test_data={'ideal_answer': "There are eight", 'behavior_name': "gemini"}, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})