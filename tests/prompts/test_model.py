
import logging
import time

from pytest import fixture
from lamoom import Lamoom, Prompt, GeminiAIModel, C_128K, AttemptToCall
logger = logging.getLogger(__name__)


@fixture
def client():
    lamoom = Lamoom()
    return lamoom
    

@fixture
def attempt():
    return AttemptToCall(
        ai_model= GeminiAIModel(
            model="blabla",
            max_tokens=C_128K                
        ),
        fallback_model=GeminiAIModel(
            model="gemini-1.5-flash",
            max_tokens=C_128K                
        )
    )

def test_loading_model(client, attempt):

    context = {
        'text': "Good morning. Tell me a funny joke!"
    }

    # initial version of the prompt
    prompt_id = f'test-{time.time()}'
    client.service.clear_cache()
    prompt = Prompt(id=prompt_id) 
    prompt.add("{text}")
    
    result = client.call(prompt.id, context, attempt)
    
    assert result.content