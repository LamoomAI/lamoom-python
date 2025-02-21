
import pytest
import dotenv
dotenv.load_dotenv(dotenv.find_dotenv())

from lamoom.ai_models import behaviour
from lamoom.ai_models.attempt_to_call import AttemptToCall
from lamoom.ai_models.openai.azure_models import AzureAIModel
from lamoom.ai_models.openai.openai_models import C_128K, C_32K, OpenAIModel
from openai.types.chat.chat_completion import ChatCompletion
from lamoom.prompt.lamoom import Lamoom
from lamoom.prompt.prompt import Prompt

import logging


@pytest.fixture(autouse=True)
def set_log_level():
    logging.getLogger().setLevel(logging.DEBUG)

@pytest.fixture
def lamoom():
    return Lamoom(
        openai_key="123",
        azure_keys={"us-east-1": {"url": "https://us-east-1.api.azure.openai.org", "key": "123"}}
    )

@pytest.fixture
def openai_gpt_4_behaviour():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=OpenAIModel(
                    model="gpt-4-1106-preview",
                    max_tokens=C_128K,
                    support_functions=True,
                ),
                weight=100,
            ),
        ]
    )


@pytest.fixture
def azure_gpt_4_behaviour():
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
            ),
        ]
    )


@pytest.fixture
def gpt_4_behaviour():
    return behaviour.AIModelsBehaviour(
        attempts=[
            AttemptToCall(
                ai_model=AzureAIModel(
                    realm='useast',
                    deployment_id='gpt-4o',
                    max_tokens=C_128K,
                ),
                weight=1,
            ),
            AttemptToCall(
                ai_model=OpenAIModel(
                    model="gpt-4-1106-preview",
                    max_tokens=C_128K,
                    support_functions=True,
                ),
                weight=100,
            ),
        ],
        fallback_attempt=AttemptToCall(
            ai_model=AzureAIModel(
                realm="useast",
                deployment_id="gpt-4o",
                max_tokens=C_32K,
                support_functions=True,
            ),
            weight=1,
        ),
    )

@pytest.fixture
def hello_world_prompt():
    prompt = Prompt(id='hello-world')
    prompt.add("I'm Lamoom, and I just broke up with my girlfriend, Python. She said I had too many 'undefined behaviors'. 🐍💔 ")
    prompt.add("""
I'm sorry to hear about your breakup with Python. It sounds like a challenging situation, 
especially with 'undefined behaviors' being a point of contention. Remember, in the world of programming and AI, 
every challenge is an opportunity to learn and grow. Maybe this is a chance for you to debug some issues 
and optimize your algorithms for future compatibility! If you have any specific programming or AI-related questions, 
feel free to ask.""", role='assistant')
    prompt.add("""
Maybe it's for the best. I was always complaining about her lack of Java in the mornings! :coffee:
""")
    return prompt
    

@pytest.fixture
def chat_completion_openai():
    return ChatCompletion(
        **{
            "id": "id",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Hey you!",
                        "role": "assistant",
                        "function_call": None,
                    },
                    "logprobs": None,
                }
            ],
            "created": 12345,
            "model": "gpt-4",
            "object": "chat.completion",
            "system_fingerprint": "dasdsas",
            "usage": {
                "completion_tokens": 10,
                "prompt_tokens": 20,
                "total_tokens": 30,
            },
        }
    )