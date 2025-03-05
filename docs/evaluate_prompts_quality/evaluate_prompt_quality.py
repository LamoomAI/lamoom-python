import logging
import random
from lamoom import Lamoom, behaviour, AttemptToCall, AzureAIModel, C_128K
from prompt import prompt_to_evaluate_prompt
from lamoom_service import get_all_prompts, get_logs
logger = logging.getLogger(__name__)



lamoom = Lamoom()

gpt4_behaviour = behaviour.AIModelsBehaviour(
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


def main():
    for prompt in get_all_prompts():
        prompt_id = prompt['prompt_id']
        prompt_chats = prompt['chats']
        logs = get_logs(prompt_id).get('items')
        if not logs or len(logs) < 5:
            continue
        contexts = []
        responses = []
        for log in random.sample(logs, 5):
            responses.append(log['response']['message'])
            contexts.append(log['context'])
        context = {
            'responses': responses,
            'prompt_data': prompt_chats,
            'prompt_id': prompt_id,
        }
        result = lamoom.call(prompt_to_evaluate_prompt.id, context, gpt4_behaviour)
        print(result.content)

if __name__ == '__main__':
    main()