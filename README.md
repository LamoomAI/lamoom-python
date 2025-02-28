# Lamoom

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

Obtain an API token from [Lamoom]('https://portal.lamoom.com') and add it as an env variable: `LAMOOM_API_TOKEN` ;

## Authentication

### Add Keys depending on models you're using:
```python
# Add LAMOOM_API_TOKEN as an environment variable:
os.setenv('LAMOOM_API_TOKEN', 'your_token_here')

# add OPENAI_API_KEY
os.setenv('OPENAI_API_KEY', 'your_key_here')

# add Azure Keys
os.setenv('AZURE_KEYS', '{"name_realm":{"url": "https://baseurl.azure.com/","key": "secret"}}')
# or creating flow_prompt obj
Lamoom(azure_keys={"realm_name":{"url": "https://baseurl.azure.com/", "key": "your_secret"}})
```

### Model Agnostic:
Mix models easily, and districute the load across models. The system will automatically distribute your load based on the weights. We support:
- Claude
- Gemini
- OpenAI (w/ Azure OpenAI models)
- Nebius with (Llama, DeepSeek, Mistral, Mixtral, dolphin, Qwen and others)

Model string format is the following for Claude, Gemini, OpenAI, Nebius:
`"{model_provider}/{model_name}"`
For Azure models format is the following:
`"azure/{realm}/{model_name}"`

```python
response_llm = client.call(agent.id, context, model = "openai/gpt-4o")
response_llm = client.call(agent.id, context, model = "azure/useast/gpt-4o")
```

### Lamoom Keys
Obtain an API token from Flow Prompt and add it:

```python
# As an environment variable:
os.setenv('LAMOOM_API_TOKEN', 'your_token_here')
# Via code: 
Lamoom(api_token='your_api_token')
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
response = client.call(prompt.id, context, "openai/gpt-4o", test_data={
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