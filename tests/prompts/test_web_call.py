
import logging
import time
from pytest import fixture
from lamoom import Lamoom, Prompt
logger = logging.getLogger(__name__)


@fixture
def client():
    import dotenv
    dotenv.load_dotenv(dotenv.find_dotenv())
    lamoom = Lamoom()
    return lamoom

def stream_function(text, **kwargs):
    print(text)

def stream_check_connection(validate, **kwargs):
    return validate


def test_web_call(client):

    context = {
        'text': "Summarize the latest reviews for the movie 'Dune: Part Two'."
    }

    # initial version of the prompt
    prompt_id = f'test-{time.time()}'
    client.service.clear_cache()
    prompt = Prompt(id=prompt_id) 
    prompt.add("{text}", role='user')
    
    # result = client.call(prompt.id, context, "openai/gpt-4o", stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})
    # result = client.call(prompt.id, context, "openai/gpt-4o")

    # result = client.call(prompt.id, context, "claude/claude-3-5-sonnet-20240620")
    
    result = client.call(prompt.id, context, "nebius/deepseek-ai/DeepSeek-R1")
    
    with open('test_web_call.txt', 'w', encoding="utf-8") as f:
        f.write(result.content)
        
    assert result.content