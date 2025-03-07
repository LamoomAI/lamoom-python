import logging
import os
import time

from pytest import fixture
from lamoom import Lamoom, Prompt
logger = logging.getLogger(__name__)


@fixture
def client():
    api_token = os.getenv("LAMOOM_API_TOKEN")
    lamoom = Lamoom(api_token=api_token)
    return lamoom


def test_creating_lamoom_test(client):
    context = {
        'ideal_answer': "There are eight planets",
        'text': "Hi! Please tell me how many planets are there in the solar system?"
    }

    # initial version of the prompt
    prompt_id = f'unit-test-creating_fp_test'
    client.service.clear_cache()
    prompt = Prompt(id=prompt_id) 
    prompt.add("{text}", role='user')

    client.create_test(prompt_id, context, model_name="gemini/gemini-1.5-flash")