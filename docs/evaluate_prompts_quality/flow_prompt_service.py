

import os
import dotenv
import requests
from lamoom.settings import LAMOOM_API_URI

dotenv.load_dotenv(dotenv.find_dotenv())

BEARER_TOKEN = os.getenv('BEARER_TOKEN')


def get_all_prompts():
    response = requests.get(f'{LAMOOM_API_URI}/prompts', headers={'Authorization': f'Bearer {BEARER_TOKEN}'})
    prompts = response.json()
    return prompts


def get_logs(prompt_id):
    response = requests.get(
        f'{LAMOOM_API_URI}/logs?prompt_id={prompt_id}&fields=response,context',
        headers={'Authorization': f'Bearer {BEARER_TOKEN}'}
    )
    logs = response.json()
    return logs

