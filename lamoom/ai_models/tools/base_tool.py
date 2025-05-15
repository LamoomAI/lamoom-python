import logging
import typing as t
from dataclasses import dataclass
import json
import re

from lamoom.utils import resolve


logger = logging.getLogger(__name__)

TOOL_CALL_NAME = 'function_call'
TOOL_CALL_RESULT_NAME = 'function_result'
# --- Constants for Prompting ---
TOOL_CALL_START_TAG = f"<{TOOL_CALL_NAME}>"
TOOL_CALL_END_TAG = f"</{TOOL_CALL_NAME}>"


def get_tool_system_prompt(tool_descriptions: str, context: t.Dict[str, str]):
    return  f"""You have next tools:
```
{resolve(tool_descriptions, context)}
```
# Tool calling procedure

Before calling any function, please follow procedure. You're doing unnecessary calls of tools. Make a mindset of what you need to do.
## 1. Think out loud what you need to do
## 2. Provide 5 whys;
## 3. Call a tool, you can call it when in <think>;

<function_call_format_rules>
If you need to use a function, use the next exactly format. That format will be parsed from your answer:
</function_call_format_rules>
```
""" + TOOL_CALL_START_TAG + """
{
"function": "...",
"parameters": {
 // parameters of the tool
}
}
""" + TOOL_CALL_END_TAG + """
```
"""


@dataclass
class ToolCallResult:
    content: str
    has_tool_call: bool
    tool_name: t.Optional[str] = None
    parameters: t.Optional[dict] = None
    execution_result: str = None


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
    param_desc = ",\n".join([f'"{p.name}": ... \\ {p.type} - ({p.description})' for p in tool.parameters])
    return f"//{tool.description}\n- {tool.name}({{{param_desc}}})"


def inject_tool_prompts(
    messages: t.List[dict],
    available_tools: t.List[ToolDefinition],
    context: t.Dict[str, str]
    ) -> t.List[dict]:
    """Injects tool descriptions and usage instructions into the system prompt."""
    if not available_tools:
        logger.debug("[inject_tool_prompts] No tools available. Returning original messages.")
        return messages

    tool_descriptions = "\n".join([format_tool_description(tool) for tool in available_tools])
    tool_system_prompt = get_tool_system_prompt(tool_descriptions, context)
    # Find system prompt or prepend to user prompt
    modified_messages = list(messages) # Create a copy
    found_system = False
    for i, msg in enumerate(modified_messages):
        if msg.get("role") == "system":
            # Append to existing system prompt
            modified_messages[i]["content"] = f"{msg.get('content', '')}\n\n{tool_system_prompt}"
            logger.debug(f"[inject_tool_prompts] Injected tool system prompt:\n{modified_messages[i]['content']}")
            found_system = True
            break

    if not found_system:
        # Prepend a new system message
        modified_messages.insert(0, {"role": "system", "content": tool_system_prompt})

        logger.debug(f"[inject_tool_prompts] Msg not found. Injected tool system prompt as a first msg :\n{tool_system_prompt}")
    return modified_messages


def parse_tool_call_block(text_response: str) -> t.Optional[ToolCallResult]:
    """
    Parses the <tool_call> block from the model's text response using regex.
    Returns a ToolCallResult object if a valid tool call is found, None otherwise.
    """
    logger.debug(f"Parsing tool call block: {text_response}")
    if not text_response:
        return None
    if not TOOL_CALL_END_TAG in text_response:
        return None
    # Regex to find the block, allowing for whitespace variations
    # DOTALL allows '.' to match newlines within the JSON block
    json_content = text_response[text_response.find(
        TOOL_CALL_START_TAG) + len(TOOL_CALL_START_TAG): text_response.rfind(TOOL_CALL_END_TAG)]
    if '```' in json_content.split('\n'):
        json_content = '\n'.join(json_content.split('\n')[1:-1])

    logger.debug(f"Found potential tool call JSON block: {json_content}")

    try:
        parsed_data = json.loads(json_content)
        # Basic validation
        if "function" in parsed_data and (
                "parameters" in parsed_data and isinstance(parsed_data["parameters"], dict) or "parameters" not in parsed_data
            ):
            logger.info(f"Successfully parsed tool call: {parsed_data['function']}")
            return ToolCallResult(
                content=text_response,
                has_tool_call=True,
                tool_name=parsed_data.get("function"),
                parameters=parsed_data.get("parameters", {}),
                execution_result=""
            )
        else:
            logger.warning(f"Parsed JSON block lacks required 'function' or 'parameters': {json_content}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from tool call block: {json_content}", exc_info=e)
        return ToolCallResult(
                content=text_response,
                has_tool_call=True,
                tool_name=None,
                parameters=None,
                execution_result=str(e)
            )


def call_function(tool_name: str, parameters: dict, tool_registry={}) -> str:
    """Handle a tool call by executing the corresponding function from the MCP registry.
    
    Args:
        tool_name: Name of the tool to execute
        parameters: Parameters for the tool
        
    Returns:
        String representation of the tool result
    """
    tool_function = tool_registry.get(tool_name)
    if not tool_function:
        logger.warning(f"Tool '{tool_name}' not found in MCP registry")
        return json.dumps({"error": f"Tool '{tool_name}' is not available."})
    tool_execution_function = tool_function.execution_function
    try:
        logger.info(f"Executing MCP tool '{tool_name}' with parameters: {parameters}")
        result = tool_execution_function(**parameters)
        logger.info(f"MCP tool '{tool_name}' executed successfully: {result}")
        return json.dumps({"result": result})
    except Exception as e:
        logger.exception(f"Error executing MCP tool '{tool_name}'", exc_info=e)
        return json.dumps({"error": f"Failed to execute tool '{tool_name}': {str(e)}"})


def handle_tool_call(current_stream_part_content, tool_registry) -> ToolCallResult:
    """Handle a tool call by executing the corresponding function from the MCP registry.
    
    Args:
        current_stream_part_content: The current content of the stream
        tool_registry: Registry of available tools
        
    Returns:
        ToolCallResult object containing the result of the tool call
    """
    parsed_tool_call = parse_tool_call_block(current_stream_part_content)
    
    if not parsed_tool_call or parsed_tool_call.error:
        return parsed_tool_call
    tool_name = parsed_tool_call.tool_name
    parameters = parsed_tool_call.parameters
    logger.info(f"Custom tool call block parsed: {tool_name}")

    # Execute the tool and get result
    tool_result_str = call_function(tool_name, parameters, tool_registry=tool_registry)
    
    return ToolCallResult(
        content=current_stream_part_content,
        has_tool_call=True,
        tool_name=tool_name,
        parameters=parameters,
        execution_result=tool_result_str,
    )


def format_tool_result_message(tool_result: ToolCallResult):
    return f'\n<{TOOL_CALL_RESULT_NAME}="{tool_result.tool_name}">\n{json.dumps(tool_result.execution_result)}\n</{TOOL_CALL_RESULT_NAME}>\n## Please, Analyze tool_call_result! Answer next using the provided response from the user'