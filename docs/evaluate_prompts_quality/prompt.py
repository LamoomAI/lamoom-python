
from lamoom import Prompt

prompt_to_evaluate_prompt = Prompt(id="prompt-improver")

prompt_to_evaluate_prompt.add(role="system", content="You're a prompt engineer, tasked with evaluating and improving prompt quality.")

prompt_to_evaluate_prompt.add(content="""The initial prompt is provided below: ```
{prompt_data}
```""", priority=1)

prompt_to_evaluate_prompt.add(
    content="{responses}", 
    is_multiple=True, 
    in_one_message=True,
    presentation="Responses to the initial prompt were as follows: ", 
    priority=2
)

prompt_to_evaluate_prompt.add(content='''
Please perform the following steps:

1. **Analyze Output Quality {prompt_id}:**
  - Examine the completeness, accuracy, and relevance of the responses.
  - Identify any common themes in errors or inaccuracies.
2. **Identify Improvement Areas:**
  - Based on the analysis, pinpoint specific areas where the prompt could be ambiguous or not detailed enough.
  - Note if the complexity of the request might be contributing to the observed output quality issues.
3. **Suggest Modifications:** 
  - Propose clear and actionable changes to the initial prompt that could potentially address the identified issues.
  - If applicable, recommend breaking down complex tasks into simpler, more manageable subtasks within the prompt.
''', required=True)
