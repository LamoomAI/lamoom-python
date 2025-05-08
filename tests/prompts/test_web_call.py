
import logging
import time
from lamoom.ai_models.tools.base_tool import TOOL_CALL_NAME
from pytest import fixture
from lamoom import Lamoom, Prompt
from lamoom.ai_models.tools.web_tool import WEB_SEARCH_TOOL
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
        'text': f"Summarize the latest reviews for the movie 'Dune: Part Two'. Please use in your answer final <{TOOL_CALL_NAME}>s"
    }

    # initial version of the prompt
    prompt_id = 'test-web-search'
    client.service.clear_cache()
    prompt = Prompt(id=prompt_id) 
    prompt.add("{text}", role='user')
    prompt.add_tool(WEB_SEARCH_TOOL)
    
    result = client.call(prompt.id, context, "openai/o4-mini")
    print(result)
    print(result.content)
    assert result.content
    with open('test_web_call_openai.txt', 'w', encoding="utf-8") as f:
        f.write(result.content)

    assert 0 == 1

    result = client.call(prompt.id, context, "claude/claude-3-5-sonnet-20240620")
    
    result = client.call(prompt.id, context, "nebius/deepseek-ai/DeepSeek-R1", stream_function=stream_function, check_connection=stream_check_connection, params={"stream": True}, stream_params={"validate": True, "end": "", "flush": True})

    with open('test_web_call.txt', 'w', encoding="utf-8") as f:
        f.write(result.content)

    assert result.content