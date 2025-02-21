# Flow Prompt

## Introduction

Lamoom is a dynamic, all-in-one library designed for managing and optimizing prompts and making tests based on the ideal answer for large language models (LLMs) in production and R&D. It facilitates dynamic data integration, latency and cost metrics visibility, and efficient load distribution across multiple AI models.

## Features

- **CI/CD testing**: Generates tests based on the context and ideal answer (usually written by the human).
- **Dynamic Prompt Development**: Avoid budget exceptions with dynamic data.
- **Multi-Model Support**: Seamlessly integrate with various LLMs like OpenAI, Anthropic, and more.
- **Real-Time Insights**: Monitor interactions, request/response metrics in production.
- **Prompt Testing and Evolution**: Quickly test and iterate on prompts using historical data.

## Installation

Install Flow Prompt using pip:

```bash
pip install lamoom
```

## Authentication

### OpenAI Keys
```python
# setting as os.env
os.setenv('OPENAI_API_KEY', 'your_key_here')
# or creating lamoom obj
Lamoom(openai_key="your_key", openai_org="your_org")
```

### Azure Keys
Add Azure keys to accommodate multiple realms:
```python
# setting as os.env
os.setenv('AZURE_KEYS', '{"name_realm":{"url": "https://baseurl.azure.com/","key": "secret"}}')
# or creating flow_prompt obj
FlowPrompt(azure_keys={"realm_name":{"url": "https://baseurl.azure.com/", "key": "your_secret"}})
```

### Model Agnostic:
Mix models easily, and districute the load across models. The system will automatically distribute your load based on the weights. We support:
- Claude
- Gemini
- OpenAI (w/ Azure OpenAI models)
- Nebius with (Llama, DeepSeek, Mistral, Mixtral, dolphin, Qwen and others)
```
from lamoom import LamoomModelProviders

def_behaviour = behaviour.AIModelsBehaviour(attempts=[
    AttemptToCall(provider='openai', model='gpt-4o', weight=100),
    AttemptToCall(provider='azure', realm='useast-1', deployment_id='gpt-4o' weight=100),
    AttemptToCall(provider='azure', realm='useast-2', deployment_id='gpt-4o' weight=100),
    AttemptToCall(provider=LamoomModelProviders.anthropic, model='claude-3-5-sonnet-20240620', weight=100
    ),
    AttemptToCall(provider=LamoomModelProviders.gemini, model='gemini-1.5-pro', weight=100
    ),
    AttemptToCall(provider=LamoomModelProviders.nebius, model='deepseek-ai/DeepSeek-R1', weight=100
    )
])

response_llm = client.call(agent.id, context, def_behaviour)
```

### Lamoom Keys
Obtain an API token from Flow Prompt and add it:

```python
# As an environment variable:
os.setenv('LAMOOM_API_TOKEN', 'your_token_here')
# Via code: 
Lamoom(api_token='your_api_token')
```

### Add Behavious:
- use OPENAI_BEHAVIOR
- or add your own Behaviour, you can set max count of attempts, if you have different AI Models, if the first attempt will fail because of retryable error, the second will be called, based on the weights.
```
from lamoom import OPENAI_GPT4_0125_PREVIEW_BEHAVIOUR
behaviour = OPENAI_GPT4_0125_PREVIEW_BEHAVIOUR
```
or:
```
from lamoom import behaviour
behaviour = behaviour.AIModelsBehaviour(
    attempts=[
        AttemptToCall(provider='azure', realm='useast-1', deployment_id='gpt-4o' weight=100),
        AttemptToCall(provider='azure', realm='useast-2', deployment_id='gpt-4o' weight=100),
    ]
)
```

## Usage Examples:

```python
from lamoom import Lamoom, Prompt

# Initialize and configure Lamoom
client = Lamoom(openai_key='your_api_key', openai_org='your_org')

# Create a prompt
prompt = Prompt('greet_user')
prompt.add("You're {name}. Say Hello and ask what's their name.", role="system")

# Call AI model with Lamoom
context = {"name": "John Doe"}
# test_data -  optional parameter used for generating tests
response = client.call(prompt.id, context, behavior, test_data={
    'ideal_answer': "Hello, I'm John Doe. What's your name?", 
    'behavior_name': "gemini"
    }
)
print(response.content)
```
- To review your created tests and score please go to https://cloud.lamoom.com/tests. You can update there Prompt and rerun tests for a published version, or saved version. If you will update and publish version online - library will automatically use the new updated version of the prompt. It's made for updating prompt without redeployment of the code, which is costly operation to do if it's required to update just prompt.

- To review logs please proceed to https://cloud.lamoom.com/logs, there you can see metrics like latency, cost, tokens;

## Best Security Practices
For production environments, it is recommended to store secrets securely and not directly in your codebase. Consider using a secret management service or encrypted environment variables.

## Contributing
We welcome contributions! Please see our Contribution Guidelines for more information on how to get involved.

## License
This project is licensed under the Apache2.0 License - see the [LICENSE](LICENSE.txt) file for details.

## Contact
For support or contributions, please contact us via GitHub Issues.