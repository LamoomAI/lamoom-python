import json
import logging
import os
import time

from pytest import fixture
from flow_prompt import FlowPrompt, behaviour, PipePrompt, AttemptToCall, NebiusAIModel, C_128K
logger = logging.getLogger(__name__)


@fixture
def fp():
    azure_keys = json.loads(os.getenv("AZURE_KEYS", "{}"))
    openai_key = os.getenv("OPENAI_API_KEY")
    api_token = os.getenv("FLOW_PROMPT_API_TOKEN")
    nebius_key = os.getenv("NEBIUS_API_KEY")
    flow_prompt = FlowPrompt(
        nebius_key=nebius_key,
        openai_key=openai_key,
        azure_keys=azure_keys,
        api_token=api_token)
    return flow_prompt


@fixture
def nebius_behaviour_new():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                provider='nebius',
                model = "deepseek-ai/DeepSeek-V3",
                max_tokens = C_128K,
                support_functions = True,
                weight=100,
            )
        ]
    )
    
@fixture
def nebius_behaviour_old():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model= NebiusAIModel(
                    model = "deepseek-ai/DeepSeek-V3",
                    max_tokens =  C_128K,
                    support_functions = True,
                ),
                weight=100,
            )
        ]
    )


def stream_function(text, **kwargs):
    logger.debug(text)

def stream_check_connection(validate, **kwargs):
    return validate

def test_nebius_behavior(fp, nebius_behaviour_new, nebius_behaviour_old):

    context = {
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }

    # initial version of the prompt
    prompt_id = f'test-{time.time()}'
    fp.service.clear_cache()
    prompt = PipePrompt(id=prompt_id) 
    prompt.add("{text}", role='user')

    response_new = fp.call(
        prompt.id, 
        context, 
        nebius_behaviour_new, 
        stream_function=stream_function, 
        check_connection=stream_check_connection, 
        params={"stream": False}, 
        stream_params={"validate": True, "end": "", "flush": True})

    response_old = fp.call(
        prompt.id, 
        context, 
        nebius_behaviour_old, 
        stream_function=stream_function, 
        check_connection=stream_check_connection, 
        params={"stream": False}, 
        stream_params={"validate": True, "end": "", "flush": True})

    logger.info(response_new.content)
    logger.info(response_old.content)
    
    assert response_new.content and response_old.content