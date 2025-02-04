
import logging
import time

from pytest import fixture
from lamoom import Lamoom, behaviour, Prompt, AttemptToCall, AzureAIModel, ClaudeAIModel, GeminiAIModel, OpenAIModel, C_128K
logger = logging.getLogger(__name__)


@fixture
def client():
    lamoom = Lamoom()
    return lamoom


@fixture
def openai_behaviour():
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
def claude_behaviour():
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
def gemini_behaviour():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=GeminiAIModel(
                    model="gemini-1.5-flash",
                    max_tokens=C_128K                
                ),
                weight=100,
            ),
        ]
    )

def stream_function(text, **kwargs):
    print(text)

def stream_check_connection(validate, **kwargs):
    return validate

def test_loading_prompt_from_service(client, openai_behaviour, claude_behaviour, gemini_behaviour):

    context = {
        'messages': ['test1', 'test2'],
        'assistant_response_in_progress': None,
        'files': ['file1', 'file2'],
        'music': ['music1', 'music2'],
        'videos': ['video1', 'video2'],
        'text': "Good morning. Tell me a funny joke!"
    }

    # initial version of the prompt
    prompt_id = f'test-{time.time()}'
    client.service.clear_cache()
    prompt = Prompt(id=prompt_id) 
    prompt.add("{text}")
    prompt.add("It's a system message, Hello {name}", role="assistant")
    
    client.call(prompt.id, context, openai_behaviour, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    client.call(prompt.id, context, claude_behaviour, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    client.call(prompt.id, context, gemini_behaviour, stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    