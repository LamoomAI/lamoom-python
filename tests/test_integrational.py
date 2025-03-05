


import logging
from time import sleep
from datetime import datetime as dt
import dotenv
dotenv.load_dotenv()
from pytest import fixture
from lamoom import Lamoom, behaviour, Prompt, AttemptToCall, AzureAIModel, C_128K
logger = logging.getLogger(__name__)


@fixture
def client():
    import dotenv
    dotenv.load_dotenv(dotenv.find_dotenv())
    lamoom = Lamoom()
    return lamoom


def test_loading_prompt_from_service(client: Lamoom):
    context = {
        'messages': ['test1', 'test2'],
        'assistant_response_in_progress': None,
        'files': ['file1', 'file2'],
        'music': ['music1', 'music2'],
        'videos': ['video1', 'video2'],
    }

    # initial version of the prompt
    prompt_id = f'unit-test-loading_prompt_from_service'
    client.service.clear_cache()
    prompt = Prompt(id=prompt_id, max_tokens=10000)
    first_str_dt = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    prompt.add(f"It's a system message, Hello at {first_str_dt}", role="system")
    prompt.add('{messages}', is_multiple=True, in_one_message=True, label='messages')
    print(client.call(prompt.id, context, "azure/useast/gpt-4o"))

    # updated version of the prompt
    client.service.clear_cache()
    prompt = Prompt(id=prompt_id, max_tokens=10000)
    next_str_dt = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    prompt.add(f"It's a system message, Hello at {next_str_dt}", role="system")
    prompt.add('{music}', is_multiple=True, in_one_message=True, label='music')
    print(client.call(prompt.id, context, "azure/useast/gpt-4o"))

    # call uses outdated version of prompt, should use updated version of the prompt
    sleep(2)
    client.service.clear_cache()
    prompt = Prompt(id=prompt_id, max_tokens=10000)
    prompt.add(f"It's a system message, Hello at {first_str_dt}", role="system")
    prompt.add('{messages}', is_multiple=True, in_one_message=True, label='messages')
    result = client.call(prompt.id, context, "azure/useast/gpt-4o")
    
    # should call the prompt with music
    assert result.prompt.messages[-1] == {'role': 'user', 'content': 'music1\nmusic2'} 
