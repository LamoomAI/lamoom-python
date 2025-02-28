
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


def test_model(client):

    context = {
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }

    # initial version of the prompt
    prompt_id = f'test-{time.time()}'
    client.service.clear_cache()
    prompt = Prompt(id=prompt_id) 
    prompt.add("{text}", role='user')

    result = client.call(prompt.id, context, "openai/o3-mini")
    
    assert result.content