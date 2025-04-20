from dataclasses import dataclass, field
import typing as t
import os
import json
import re
import logging
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ID = os.getenv("SEARCH_ENGINE_ID")

# --- Constants for Prompting ---
TOOL_CALL_START_TAG = "<tool_call>"
TOOL_CALL_END_TAG = "</tool_call>"
TOOL_PROMPT_HEADER = """
You have the following skills available:"""

TOOL_PROMPT_FOOTER = f"""
If you determine you need to use one of your skills to fulfill the user's request, you MUST format your request as follows, replacing the placeholders:
{TOOL_CALL_START_TAG}
{{
  "tool_name": "<name_of_the_tool_to_use>",
  "parameters": {{
    "<parameter_name_1>": <parameter_value_1>,
    "<parameter_name_2>": <parameter_value_2>
    // ... include all required parameters for the chosen tool
  }}
}}
{TOOL_CALL_END_TAG}
Only include the {TOOL_CALL_START_TAG}...{TOOL_CALL_END_TAG} block if you need to use a skill. Do not add any other text before or after the block if you decide to use a skill."""

@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: t.List[ToolParameter]
    execution_function: t.Callable
    

def format_tool_description(tool: ToolDefinition) -> str:
    """Formats a single tool's description for the prompt."""
    param_desc = ", ".join([f"{p.name}: {p.type} ({p.description})" for p in tool.parameters])
    return f"- {tool.name}({{{param_desc}}}) - {tool.description}"

def inject_tool_prompts(
    messages: t.List[dict],
    available_tools: t.List[ToolDefinition]
    ) -> t.List[dict]:
    """Injects tool descriptions and usage instructions into the system prompt."""
    if not available_tools:
        return messages

    tool_descriptions = "\n".join([format_tool_description(tool) for tool in available_tools])
    tool_system_prompt = f"{TOOL_PROMPT_HEADER}\n{tool_descriptions}\n{TOOL_PROMPT_FOOTER}"

    # Find system prompt or prepend to user prompt
    modified_messages = list(messages) # Create a copy
    found_system = False
    for i, msg in enumerate(modified_messages):
        if msg.get("role") == "system":
            # Append to existing system prompt
            modified_messages[i]["content"] = f"{msg.get('content', '')}\n\n{tool_system_prompt}"
            found_system = True
            break

    if not found_system:
        # Prepend a new system message
        modified_messages.insert(0, {"role": "system", "content": tool_system_prompt})

    logger.debug(f"Injected tool system prompt:\n{tool_system_prompt}")
    return modified_messages

def parse_tool_call_block(text_response: str) -> t.Optional[t.Dict[str, t.Any]]:
    """
    Parses the <tool_call> block from the model's text response using regex.
    Returns the parsed JSON content as a dict, or None if not found or invalid.
    """
    if not text_response:
        return None

    # Regex to find the block, allowing for whitespace variations
    # DOTALL allows '.' to match newlines within the JSON block
    match = re.search(
        rf"{re.escape(TOOL_CALL_START_TAG)}(.*?){re.escape(TOOL_CALL_END_TAG)}",
        text_response,
        re.DOTALL | re.IGNORECASE
    )

    if not match:
        return None

    json_content = match.group(1).strip()
    logger.debug(f"Found potential tool call JSON block: {json_content}")

    try:
        parsed_data = json.loads(json_content)
        # Basic validation
        if "tool_name" in parsed_data and "parameters" in parsed_data and isinstance(parsed_data["parameters"], dict):
            logger.info(f"Successfully parsed tool call: {parsed_data['tool_name']}")
            return parsed_data
        else:
            logger.warning(f"Parsed JSON block lacks required 'tool_name' or 'parameters': {json_content}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from tool call block: {json_content}", exc_info=e)
        return None
    
def format_tool_result_message(tool_name: str, tool_result: str) -> dict:
    """Formats the tool execution result into a message for the history."""
    return {
        "role": "user",
        "content": f"Tool execution result for '{tool_name}':\n{tool_result}"
    }

def scrape_webpage(url: str):
    """
    Scrapes the content of a webpage and returns the text.
    """
    if not url.startswith(("https://", "http://")):
        url = "https://" + url
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        clean_text = text.splitlines()
        clean_text = [element.strip()
                      for element in clean_text if element.strip()]
        clean_text = '\n'.join(clean_text)
        return clean_text

    else:
        return "Failed to retrieve the website content."


def perform_web_search(query: str):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': API_KEY,
        'cx': SEARCH_ID,
        'num': 3
    }

    response = requests.get(url, params=params)
    results = response.json()

    search_result = ""
    
    if 'items' in results:
        for result in results['items']:
            search_result += scrape_webpage(result['link']) + '\n'
    
    return search_result

web_call_tool = ToolDefinition(
    name="web_call",
    description="Performs a web search using a search engine to find up-to-date information or details not present in the internal knowledge.",
    parameters=[
        ToolParameter(name="query", type="string", description="The search query to use.", required=True)
    ],
    execution_function=perform_web_search
)

AVAILABLE_TOOLS_REGISTRY: t.Dict[str, ToolDefinition] = {
    "web_call": web_call_tool,
}