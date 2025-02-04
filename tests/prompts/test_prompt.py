

from lamoom.ai_models.attempt_to_call import AttemptToCall
from lamoom.ai_models.openai.azure_models import AzureAIModel
from lamoom.ai_models.openai.openai_models import C_128K
from lamoom.exceptions import NotEnoughBudgetError
from lamoom.prompt.prompt import Prompt

import pytest


@pytest.fixture
def azure_ai_attempt():
    return AttemptToCall(
        ai_model=AzureAIModel(
            realm='us-east-1',
            deployment_id="gpt-4-1106-preview",
            max_tokens=C_128K,
            support_functions=True,
        ),
        weight=100,
    )

def test_load_dump_prompt():
    prompt = Prompt(id='hello-world', max_tokens=100)
    prompt.add("I'm Lamoom, and I just broke up with my girlfriend, Python. She said I had too many 'undefined behaviors'. 🐍💔 ")
    prompt.add("""
I'm sorry to hear about your breakup with Python. It sounds like a challenging situation, 
especially with 'undefined behaviors' being a point of contention. Remember, in the world of programming and AI, 
every challenge is an opportunity to learn and grow. Maybe this is a chance for you to debug some issues 
and optimize your algorithms for future compatibility! If you have any specific programming or AI-related questions, 
feel free to ask.""", role='assistant', priority=2)
    prompt.add("""
Maybe it's for the best. I was always complaining about her lack of Java in the mornings! :coffee:
""")
    loaded_prompt = Prompt.load(prompt.dump())
    assert prompt.dump() == loaded_prompt.dump()


def test_prompt_add(azure_ai_attempt: AttemptToCall):
    pipe = Prompt(id='test')
    pipe.add("Hello, how can I help you today?")
    uer_prompt = pipe.create_prompt(azure_ai_attempt)
    assert len(uer_prompt.pipe) == 1
    assert uer_prompt.priorities[0][0].content == "Hello, how can I help you today?"


def test_prompt_initialize(azure_ai_attempt: AttemptToCall):
    pipe = Prompt(id='test')
    user_prompt = pipe.create_prompt(azure_ai_attempt)

    user_prompt.add("Hello, how can I help you today?")

    context = {}
    initialized_pipe = user_prompt.resolve(context)
    messages = initialized_pipe.messages
    assert len(messages) == 1
    assert messages[0].content == "Hello, how can I help you today?"


def test_prompt_initialize_not_enough_budget(azure_ai_attempt:  AttemptToCall):
    pipe = Prompt(id='test')
    user_prompt = pipe.create_prompt(azure_ai_attempt)
    user_prompt.add("Hello, how can I help you today?", required=True)

    context = {}
    user_prompt.min_sample_tokens = 1299  # Not enough tokens for the message
    user_prompt.model_max_tokens = 1300  # Not enough tokens for the message
    with pytest.raises(NotEnoughBudgetError):
        user_prompt.resolve(context)


def test_prompt_show_pipe():
    pipe = Prompt(id='test')
    pipe.add("Hello, how can I help you today?")
    pipe_dump = pipe.dump()
    assert pipe_dump['id'] == 'test'
    assert pipe_dump['max_tokens'] is None
    assert pipe_dump['min_sample_tokens'] == 3000
    assert pipe_dump['reserved_tokens_budget_for_sampling'] is None
    assert len(pipe_dump['pipe']) == 1
    assert pipe_dump['priorities'] == {0: [{'content': 'Hello, how can I help you today?', 'role': 'user', 'priority': 0, 'required': False, 'is_multiple': False, 'while_fits': False, 'add_in_reverse_order': False, 'in_one_message': False, 'continue_if_doesnt_fit': False}]}

def test_prompt_left_budget(azure_ai_attempt:  AttemptToCall):
    pipe = Prompt(id='test')
    pipe.add("Hello, how can I help you today?")
    user_prompt = pipe.create_prompt(azure_ai_attempt)
    user_prompt.model_max_tokens = 2030
    user_prompt.reserved_tokens_budget_for_sampling = 2030
    initialized_pipe = user_prompt.resolve({})
    assert (
        initialized_pipe.left_budget
        == user_prompt.model_max_tokens
        - initialized_pipe.prompt_budget
        - user_prompt.safe_gap_tokens
    )


def test_prompt_prompt_price(azure_ai_attempt:  AttemptToCall):
    pipe = Prompt(id='test')
    pipe.add("Hello, how can I help you today?")
    user_prompt = pipe.create_prompt(azure_ai_attempt)
    user_prompt.model_max_tokens = 4030
    user_prompt.add("Hello " + 'world ' * 1000)
    pipe = user_prompt.resolve({})
    assert len(pipe.get_messages()) == 1


def test_prompt_calculate_budget_for_values(azure_ai_attempt:  AttemptToCall):
    pipe = Prompt(id='test')
    pipe.max_tokens = 1400
    pipe.min_sample_tokens = 1000

    pipe.add("Priority. Hello {name}", priority=1)
    pipe.add("2d priority. Hello {name}", priority=2)
    pipe.add(
        "no priority. didn't fit. Hello {name}" + ("hello" * 1000), priority=2
    )
    user_prompt = pipe.create_prompt(azure_ai_attempt)
    prompt = user_prompt.resolve({"name": "World"})
    messages = prompt.get_messages()
    assert len(messages) == 2
    assert messages[0]["content"] == "Priority. Hello World"
    assert messages[1]["content"] == "2d priority. Hello World"


def test_prompt_copy():
    pipe = Prompt(id='test')
    pipe.add("Hello, how can I help you today?")
    copy = pipe.copy('new_id')
    assert copy.id == 'new_id'
    copy_dump = copy.dump()
    assert copy_dump['id'] == 'new_id'
    original_dump = pipe.dump()
    original_dump.pop('id')
    copy_dump.pop('id')
    assert original_dump == copy_dump
