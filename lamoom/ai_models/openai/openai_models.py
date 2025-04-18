import logging
import typing as t
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from openai import OpenAI

from lamoom.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
from lamoom.ai_models.constants import C_128K, C_16K, C_32K, C_4K
from lamoom.ai_models.openai.responses import OpenAIResponse
from lamoom.ai_models.utils import get_common_args
from lamoom.exceptions import ConnectionLostError
from lamoom.ai_models.tools import ToolDefinition, AVAILABLE_TOOLS
import json

from openai.types.chat import ChatCompletionMessage as Message
from lamoom.responses import Prompt

from .utils import raise_openai_exception

M_DAVINCI = "davinci"

logger = logging.getLogger(__name__)


class FamilyModel(Enum):
    chat = "GPT-3.5"
    gpt4 = "GPT-4"
    gpt4o = "GPT-4o"
    gpt4o_mini = "GPT-4o-mini"
    instruct_gpt = "InstructGPT"

BASE_URL_MAPPING = {
    'gemini': "https://generativelanguage.googleapis.com/v1beta/openai/",
    'nebius': 'https://api.studio.nebius.ai/v1/'
}


@dataclass(kw_only=True)
class OpenAIModel(AIModel):
    model: t.Optional[str]
    max_tokens: int = C_16K
    support_functions: bool = False
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.OPENAI
    family: str = None
    max_sample_budget: int = C_4K
    base_url: str = None

    def __str__(self) -> str:
        return f"openai-{self.model}-{self.family}"

    def __post_init__(self):
        if self.model.startswith("davinci"):
            self.family = FamilyModel.instruct_gpt.value
        elif self.model.startswith("gpt-3"):
            self.family = FamilyModel.chat.value
        elif self.model.startswith("gpt-4o-mini"):
            self.family = FamilyModel.gpt4o_mini.value
        elif self.model.startswith("gpt-4o"):
            self.family = FamilyModel.gpt4o.value
        elif self.model.startswith(("gpt4", "gpt-4", "gpt")):
            self.family = FamilyModel.gpt4.value
        else:
            logger.info(
                f"Unknown family for {self.model}. Please add it obviously. Setting as GPT4"
            )
            self.family = FamilyModel.gpt4.value
        logger.debug(f"Initialized OpenAIModel: {self}")

    @property
    def name(self) -> str:
        return self.model

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
        }

    def get_base_url(self) -> str | None:
        return BASE_URL_MAPPING.get(self.provider.value, None)
    
    def get_metrics_data(self):
        return {
            "model": self.model,
            "family": self.family,
            "provider": self.provider.value,
            "base_url": self.get_base_url() if self.base_url is None else self.base_url
        }

    def call(
        self,
        messages,
        max_tokens,
        stream_function: t.Callable = None,
        check_connection: t.Callable = None,
        stream_params: dict = {},
        client_secrets: dict = {},
        **kwargs,
    ) -> OpenAIResponse:
        logger.debug(
            f"Calling {messages} with max_tokens {max_tokens} and kwargs {kwargs}"
        )
        if self.family in [
            FamilyModel.chat.value,
            FamilyModel.gpt4.value,
            FamilyModel.gpt4o.value,
            FamilyModel.gpt4o_mini.value,
        ]:
            return self.call_chat_completion(
                messages,
                max_tokens,
                stream_function=stream_function,
                check_connection=check_connection,
                stream_params=stream_params,
                client_secrets=client_secrets,
                **kwargs,
            )
        raise NotImplementedError(f"Openai family {self.family} is not implemented")

    def get_client(self, client_secrets: dict = {}):
        return OpenAI(
            organization=client_secrets.get("organization", None),
            api_key=client_secrets["api_key"],
            base_url=self.get_base_url() if self.base_url is None else self.base_url
        )
        
    def _format_tools_for_openai(self, tools: t.List[ToolDefinition]) -> t.List[t.Dict[str, t.Any]]:
        openai_tools = []
        for tool_def in tools:
            properties = {}
            required_params = []
            for param in tool_def.parameters:
                # Basic type mapping, expand as needed
                param_type = "string" # Default or map more types
                if param.type == "number":
                    param_type = "number"
                elif param.type == "boolean":
                    param_type = "boolean"

                properties[param.name] = {
                    "type": param_type,
                    "description": param.description,
                }
                if param.required:
                    required_params.append(param.name)

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required_params,
                    },
                },
            })
        return openai_tools

    def call_chat_completion(
        self,
        messages: t.List[t.Dict[str, str]],
        max_tokens: t.Optional[int],
        functions: t.List[t.Dict[str, str]] = [],
        available_tools: t.List[ToolDefinition] = AVAILABLE_TOOLS,
        stream_function: t.Callable = None,
        check_connection: t.Callable = None,
        stream_params: dict = {},
        client_secrets: dict = {},
        **kwargs,
    ) -> OpenAIResponse:
                
        client = self.get_client(client_secrets)
        
        # Keep track of the conversation history including tool interactions
        current_messages = list(messages)
        current_messages.insert(0, {"role": "system", "content": "Please use a web_call function if you don't have the answer in your knowledge base."})
        
        while True:
            # if functions:
            #     kwargs["tools"] = functions
            call_kwargs = {
                **{
                    "messages": current_messages,
                },
                **self.get_params(),
                **kwargs,
            }
            
            # Format and add tools if available AND if the model supports them
            openai_formatted_tools = []
            if available_tools:
                openai_formatted_tools = self._format_tools_for_openai(available_tools)
                
                logger.info(f"TOOLS: {openai_formatted_tools}")
                
                if openai_formatted_tools:
                    call_kwargs["tools"] = openai_formatted_tools
            
            try:
                result = client.chat.completions.create(
                    **call_kwargs,
                )

                #TODO: handle streaming part later
                if kwargs.get("stream"):
                    return OpenAIStreamResponse(
                        stream_function=stream_function,
                        check_connection=check_connection,
                        stream_params=stream_params,
                        available_tools=available_tools,
                        original_result=result,
                        initial_call_kwargs=call_kwargs,
                        client=client,
                        prompt=Prompt(
                            messages=kwargs.get("messages"),
                            functions=kwargs.get("tools"),
                            max_tokens=max_tokens,
                            temperature=kwargs.get("temperature"),
                            top_p=kwargs.get("top_p"),
                        ),
                    ).stream()
                    
                logger.debug(f"Result: {result.choices[0]}")
                choice = result.choices[0]
                response_message = choice.message
                # *** TOOL CALL CHECK ***
                
                if response_message.tool_calls:
                    logger.info(f"Tool call requested: {response_message.tool_calls}")
                    # Append the assistant's message asking for the tool call
                    current_messages.append(response_message.to_dict()) # Or convert appropriately

                    # Execute tools
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            for tool in available_tools:
                                if tool.name == function_name:
                                    tool_definition = tool
                                    break

                            if tool_definition and tool_definition.execution_function:
                                logger.info(f"Executing tool: {function_name} with args: {function_args}")
                                # *** EXECUTE THE ACTUAL TOOL FUNCTION ***
                                tool_result = tool_definition.execution_function(**function_args)
                                logger.info(f"Tool result: {tool_result[:200]}...") # Log truncated result
                            else:
                                logger.warning(f"Tool '{function_name}' not found or not executable.")
                                tool_result = json.dumps({"status": "error", "message": f"Tool '{function_name}' not implemented."})

                            # Append the tool result message
                            current_messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "content": tool_result,
                                }
                            )
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode arguments for tool {function_name}: {tool_call.function.arguments}")
                            # Append an error message back
                            current_messages.append({
                                "tool_call_id": tool_call.id, "role": "tool", "name": function_name,
                                "content": json.dumps({"status": "error", "message": "Invalid arguments format."})
                            })
                        except Exception as tool_exc:
                            logger.exception(f"Error executing tool {function_name}", exc_info=tool_exc)
                            current_messages.append({
                                "tool_call_id": tool_call.id, "role": "tool", "name": function_name,
                                "content": json.dumps({"status": "error", "message": f"Execution failed: {tool_exc}"})
                            })

                    # *Loop back* to call the model again with the tool results included
                    # (Remove tools for the next call if you only want one round of tool use per turn,
                    # or keep them if you want multi-step tool chains)
                    # call_kwargs.pop("tools", None)
                    # call_kwargs.pop("tool_choice", None)
                    continue # Go to the start of the while loop

                # *** NO TOOL CALL - Normal Response ***
                else:
                    logger.debug(f"Final response message: {response_message}")
                    return OpenAIResponse(
                        finish_reason=choice.finish_reason,
                        message=response_message,
                        content=response_message.content,
                        original_result=result,
                        prompt=Prompt(
                            messages=messages, # Original messages
                            functions=call_kwargs.get("tools"), # The tools that were available
                            max_tokens=max_tokens,
                            temperature=kwargs.get("temperature"),
                            top_p=kwargs.get("top_p"),
                        ),
                    )
            except Exception as e:
                logger.exception("[OPENAI] failed to handle chat stream", exc_info=e)
                raise_openai_exception(e)


@dataclass(kw_only=True)
class OpenAIStreamResponse(OpenAIResponse):
    # Existing fields
    stream_function: t.Callable
    check_connection: t.Callable
    stream_params: dict

    # Add context needed for tool calls within streaming (as per previous plan)
    client: OpenAI # The OpenAI client instance
    initial_call_kwargs: dict # Kwargs used for the first call (mutable copy needed)
    available_tools: t.List[ToolDefinition]

    # Override stream method with tool handling logic
    def stream(self):
        """
        Processes the stream, handles potential tool calls by executing tools
        and restarting the stream, yields final content via stream_function.
        Returns the final populated OpenAIStreamResponse object.
        """
        tool_registry = {} # Dictionary to hold tool definitions for easy access
        # Populate the tool registry with available tools
        for tool in self.available_tools:
            tool_registry[tool.name] = tool
            
        final_content_accumulator = "" # Accumulates text content across stream restarts
        current_messages_history = list(self.initial_call_kwargs["messages"]) # Use mutable copy of history

        stream_iterator = self.original_result # Start with the initial stream

        outer_loop_iteration = 0 # To prevent potential infinite loops
        max_tool_iterations = 5 # Safety break for sequential tool calls

        while outer_loop_iteration < max_tool_iterations:
            outer_loop_iteration += 1
            logger.debug(f"--- Starting stream processing loop iteration: {outer_loop_iteration} ---")

            current_chunk_content = "" # Content accumulated in this specific stream part
            tool_calls_accumulator = [] # Buffer for assembling tool call info for this stream part
            assistant_message_for_history = None # To store the assistant message asking for tools
            current_finish_reason = None # Finish reason for the current stream part

            try:
                logger.debug("Processing stream chunks...")
                for stream_index, chunk in enumerate(stream_iterator):
                    if not chunk.choices:
                        logger.debug("Skipping chunk with no choices.")
                        continue

                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason
                    current_finish_reason = finish_reason # Track the latest finish reason

                    # --- 1. Accumulate Text Content & Stream Out ---
                    if delta and delta.content:
                        text_chunk = delta.content
                        current_chunk_content += text_chunk
                        final_content_accumulator += text_chunk
                        # Use your existing process_message for streaming out and connection checks
                        self.process_message(text_chunk, stream_index)

                    # --- 2. Accumulate Tool Call Information ---
                    if delta and delta.tool_calls:
                        # This part handles assembling potentially chunked tool call data
                        for tool_call_chunk in delta.tool_calls:
                            index = tool_call_chunk.index
                            # Ensure list is long enough
                            while len(tool_calls_accumulator) <= index:
                                tool_calls_accumulator.append(
                                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                                )

                            # Update existing entry
                            if tool_call_chunk.id:
                                tool_calls_accumulator[index]["id"] = tool_call_chunk.id
                            if tool_call_chunk.function:
                                if tool_call_chunk.function.name:
                                    tool_calls_accumulator[index]["function"]["name"] += tool_call_chunk.function.name
                                if tool_call_chunk.function.arguments:
                                    tool_calls_accumulator[index]["function"]["arguments"] += tool_call_chunk.function.arguments

                    # --- Capture Assistant Role for History ---
                    # Store the basic structure if we haven't yet and it's an assistant delta
                    # This helps build the message to add to history if tool calls occur
                    if delta and delta.role == "assistant" and not assistant_message_for_history:
                         assistant_message_for_history = {"role": "assistant", "content": None} # Start with no content

                    # --- Check if stream part finished ---
                    if finish_reason:
                        logger.debug(f"Stream part finished with reason: {finish_reason}")
                        break # Exit inner loop (chunk processing)

                # --- End of inner chunk processing loop for this stream part ---

                # --- 3. Handle Finish Reason: Tool Calls ---
                if current_finish_reason == "tool_calls":
                    logger.info(f"Tool call(s) requested by model. Assembled calls: {json.dumps(tool_calls_accumulator)}")

                    if not assistant_message_for_history:
                        # Should have been set by a delta, but add fallback
                         assistant_message_for_history = {"role": "assistant", "content": None}

                    # Add the assembled tool calls to the assistant message
                    assistant_message_for_history["tool_calls"] = tool_calls_accumulator

                    # Add the assistant's request to the conversation history
                    current_messages_history.append(assistant_message_for_history)

                    # --- 4. Execute Tools ---
                    tool_results_messages = []
                    for tool_call_data in tool_calls_accumulator:
                        tool_call_id = tool_call_data.get("id")
                        function_info = tool_call_data.get("function", {})
                        function_name = function_info.get("name")
                        function_args_str = function_info.get("arguments", "{}")
                        tool_result_content = ""

                        if not tool_call_id or not function_name:
                             logger.error(f"Incomplete tool call data received: {tool_call_data}")
                             tool_result_content = json.dumps({"status": "error", "message": "Incomplete tool call data from model."})
                        else:
                            tool_definition = tool_registry.get(function_name)
                            if tool_definition and tool_definition.execution_function:
                                try:
                                    logger.info(f"Attempting to parse args for {function_name}: {function_args_str}")
                                    function_args = json.loads(function_args_str)
                                    logger.info(f"Executing tool: '{function_name}' with args: {function_args}")

                                    # *** EXECUTE THE ACTUAL TOOL FUNCTION ***
                                    tool_result = tool_definition.execution_function(**function_args)
                                    # Ensure result is a string (as required by API)
                                    if not isinstance(tool_result, str):
                                        tool_result_content = json.dumps(tool_result)
                                    else:
                                        tool_result_content = tool_result
                                    logger.info(f"Tool '{function_name}' executed successfully. Result snippet: {tool_result_content[:200]}...")

                                except json.JSONDecodeError:
                                    logger.error(f"Failed to decode JSON arguments for tool '{function_name}': {function_args_str}")
                                    tool_result_content = json.dumps({"status": "error", "message": f"Invalid JSON arguments provided for {function_name}."})
                                except Exception as tool_exc:
                                    logger.exception(f"Error executing tool '{function_name}'", exc_info=tool_exc)
                                    tool_result_content = json.dumps({"status": "error", "message": f"Execution of tool '{function_name}' failed: {str(tool_exc)}"})
                            else:
                                logger.warning(f"Tool '{function_name}' requested by model but not found in registry or not executable.")
                                tool_result_content = json.dumps({"status": "error", "message": f"Tool '{function_name}' is not available or implemented."})

                        # Append the tool result message for the next API call
                        tool_results_messages.append({
                            "tool_call_id": tool_call_id,
                            "role": "tool",
                            "name": function_name,
                            "content": tool_result_content, # Result MUST be a string
                        })

                    # Add all tool results to the history
                    current_messages_history.extend(tool_results_messages)

                    # --- 5. Make a *new* streaming call ---
                    logger.info("Restarting stream after processing tool calls.")
                    new_call_kwargs = {
                        **self.initial_call_kwargs, # Carry over original params
                        "messages": current_messages_history, # Use updated message history
                        "stream": True, # MUST be true
                        # Keep tools available for potential multi-turn calls
                        "tools": self.initial_call_kwargs.get("tools"),
                    }

                    # Update the stream iterator for the outer loop
                    stream_iterator = self.client.chat.completions.create(**new_call_kwargs)

                    # Continue to the next iteration of the outer while loop
                    # to process the *new* stream
                    continue # Goes back to `while outer_loop_iteration < max_tool_iterations:`

                # --- 6. Handle Finish Reason: Stop or Length (Normal Termination) ---
                elif current_finish_reason == "stop" or current_finish_reason == "length":
                    logger.info(f"Final stream finished with reason: {current_finish_reason}")
                    # The stream has ended successfully (possibly after tool calls)
                    self.finish_reason = current_finish_reason
                    # Construct the final message object based on accumulated content
                    self.message = Message(
                        content=final_content_accumulator, # All content from all parts
                        role="assistant"
                        )
                    self.content = final_content_accumulator
                    # We've reached a terminal state, break the outer loop
                    break

                # --- 7. Handle Other/Unexpected Finish Reasons or End of Stream ---
                else:
                    # This might happen if the stream ends unexpectedly or with an unknown reason
                    logger.warning(f"Stream ended with unexpected or no finish_reason: {current_finish_reason}. Last chunk content: {current_chunk_content}")
                    self.finish_reason = current_finish_reason or "unknown"
                    self.message = Message(content=final_content_accumulator, role="assistant")
                    self.content = final_content_accumulator
                    break # Exit outer loop

            # --- Exception Handling for the inner loop ---
            except ConnectionLostError as cle:
                 logger.error("Connection lost during stream processing.", exc_info=cle)
                 self.finish_reason = "error_connection_lost"
                 self.message = Message(content=final_content_accumulator, role="assistant")
                 self.content = final_content_accumulator
                 raise cle # Re-raise specific error
            except Exception as e:
                logger.exception("Exception during stream chunk processing", exc_info=e)
                self.finish_reason = "error_processing_stream"
                self.message = Message(content=final_content_accumulator, role="assistant")
                self.content = final_content_accumulator
                # Use your specific exception raising mechanism
                raise_openai_exception(e) # This might terminate the process

        # --- End of outer while loop ---

        if outer_loop_iteration >= max_tool_iterations:
             logger.warning(f"Reached maximum tool call iterations ({max_tool_iterations}). Stopping.")
             self.finish_reason = "error_max_tool_iterations"
             # Final message might be incomplete if loop was broken
             if not self.message:
                 self.message = Message(content=final_content_accumulator, role="assistant")
                 self.content = final_content_accumulator

        # Return self instance with populated data (content, message, finish_reason)
        logger.debug(f"Stream processing complete. Final finish reason: {self.finish_reason}")
        with open('kwargs_stream.txt', 'w', encoding="utf-8") as f:
            f.write(str(new_call_kwargs))
        return self

    def process_message(self, text: str, idx: int):
        if idx % 5 == 0:
            if not self.check_connection(**self.stream_params):
                raise ConnectionLostError("Connection was lost!")
        if not text:
            return
        self.stream_function(text, **self.stream_params)

    # def stream(self):
    #     content = ""
    #     for i, data in enumerate(self.original_result):
    #         if not data.choices:
    #             continue
    #         choice = data.choices[0]
    #         if choice.delta:
    #             content += choice.delta.content or ""
    #             self.process_message(choice.delta.content, i)
    #     self.message = Message(
    #         content=content,
    #         role="assistant",
    #     )
    #     return self
