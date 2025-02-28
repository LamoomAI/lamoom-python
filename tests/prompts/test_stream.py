
import logging
import time

from pytest import fixture
from lamoom import Lamoom, Prompt, OpenAIResponse
logger = logging.getLogger(__name__)


@fixture
def client():
    lamoom = Lamoom()
    return lamoom

def stream_function(text, **kwargs):
    print(text)

def stream_check_connection(validate, **kwargs):
    return validate

def test_stream(client: Lamoom):

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
    
    result: OpenAIResponse = client.call(prompt.id, context, "azure/useast/gpt-4o", stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    client.call(prompt.id, context, "claude/claude-3-haiku-20240307", stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    client.call(prompt.id, context, "gemini/gemini-1.5-flash", stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    
    assert "message" in result.to_dict()