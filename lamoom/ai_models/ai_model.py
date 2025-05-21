import json
import typing as t
from dataclasses import dataclass, field
from enum import Enum
import logging
from _decimal import Decimal

import tiktoken

from lamoom import settings
from lamoom.ai_models.tools.base_tool import ToolCallResult, ToolDefinition, parse_tool_call_block
from lamoom.responses import AIResponse, StreamingResponse
from lamoom.exceptions import RetryableCustomError, StopStreamingError
from lamoom.utils import current_timestamp_ms

logger = logging.getLogger(__name__)

class AI_MODELS_PROVIDER(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    CLAUDE = "claude"
    GEMINI = "gemini"
    CUSTOM = "custom"

    def is_custom(self):
        return self == AI_MODELS_PROVIDER.CUSTOM


encoding = tiktoken.get_encoding("cl100k_base")

class TagParser:
    """Parser for handling streaming content with ignore tags."""
    MAX_BUFFER_SIZE = 50

    def __init__(self, ignore_tags: t.List[str] = None):
        self.ignore_tags = set(ignore_tags or [])
        self.reset()

    def reset(self):
        self.state = {
            'buffer': '',
            'in_ignored_tag': False,
            'ignored_tag': None,
            'partial_tag_buffer': '',
        }

    def _is_partial_ignored_tag(self, buffer: str) -> bool:
        if not buffer.startswith('<'):
            return False
        for tag in self.ignore_tags:
            if tag.startswith(buffer[1:]):
                return True
        return False

    def _is_valid_tag(self, tag: str) -> bool:
        if not (tag.startswith('<') and tag.endswith('>')):
            return False
        tag_content = tag[1:-1].strip()
        if tag_content.startswith('/'):
            return len(tag_content) > 1
        return len(tag_content) > 0

    def text_to_stream_chunk(self, chunk: str) -> str:
        return chunk
        logger = logging.getLogger("TagParser")
        logger.debug(f"[INPUT] chunk: {chunk!r}")

        # Always process buffer first if it exists
        if self.state['buffer']:
            logger.debug(f"[BUFFER] Prepending buffer: {self.state['buffer']!r} to chunk: {chunk!r}")
            chunk = self.state['buffer'] + chunk
            self.state['buffer'] = ''

        # If we're in an ignored tag, just buffer everything
        if self.state['in_ignored_tag']:
            self.state['buffer'] += chunk
            close_tag = f'</{self.state["ignored_tag"]}>'
            idx = self.state['buffer'].find(close_tag)
            if idx == -1:
                if len(self.state['buffer']) > self.MAX_BUFFER_SIZE:
                    self.state['buffer'] = self.state['buffer'][-self.MAX_BUFFER_SIZE:]
                logger.debug(f"[IGNORED] Still in ignored tag: {self.state['ignored_tag']!r}, buffer: {self.state['buffer']!r}")
                return ''
            self.state['buffer'] = self.state['buffer'][idx + len(close_tag):]
            logger.debug(f"[IGNORED] Closed ignored tag: {self.state['ignored_tag']!r}")
            self.state['in_ignored_tag'] = False
            self.state['ignored_tag'] = None
            # After closing ignored tag, process the rest of the buffer in the next call
            return ''

        # Handle partial tag buffer
        if self.state['partial_tag_buffer']:
            self.state['partial_tag_buffer'] += chunk
            if '\n' in self.state['partial_tag_buffer']:
                result = self.state['partial_tag_buffer']
                self.state['partial_tag_buffer'] = ''
                logger.debug(f"[PARTIAL] Newline in partial tag buffer, output: {result!r}")
                return result
            if '>' in self.state['partial_tag_buffer']:
                gt_idx = self.state['partial_tag_buffer'].find('>')
                tag = self.state['partial_tag_buffer'][:gt_idx + 1]
                rest = self.state['partial_tag_buffer'][gt_idx + 1:]
                if self._is_valid_tag(tag):
                    tag_name = tag[1:-1].strip().split()[0]
                    if tag_name in self.ignore_tags:
                        self.state['in_ignored_tag'] = True
                        self.state['ignored_tag'] = tag_name
                        self.state['partial_tag_buffer'] = ''
                        logger.debug(f"[PARTIAL] Entered ignored tag: {tag_name!r}")
                        return ''
                    else:
                        result = self.state['partial_tag_buffer']
                        self.state['partial_tag_buffer'] = ''
                        logger.debug(f"[PARTIAL] Outputting non-ignored tag: {result!r}")
                        return result
                else:
                    result = self.state['partial_tag_buffer']
                    self.state['partial_tag_buffer'] = ''
                    logger.debug(f"[PARTIAL] Outputting invalid tag: {result!r}")
                    return result
            if self._is_partial_ignored_tag(self.state['partial_tag_buffer']):
                logger.debug(f"[PARTIAL] Buffering possible ignored tag: {self.state['partial_tag_buffer']!r}")
                return ''
            else:
                result = self.state['partial_tag_buffer']
                self.state['partial_tag_buffer'] = ''
                logger.debug(f"[PARTIAL] Outputting not-ignored tag: {result!r}")
                return result

        # Main logic: flush up to the first ignored tag, then stop processing further in this call
        output = ''
        i = 0
        while i < len(chunk):
            c = chunk[i]
            if c == '<':
                # Check if this could be the start of an ignored tag
                for tag in self.ignore_tags:
                    if chunk[i+1:i+1+len(tag)] == tag:
                        # Found start of ignored tag
                        self.state['in_ignored_tag'] = True
                        self.state['ignored_tag'] = tag
                        self.state['buffer'] = chunk[i:]  # Buffer the rest for next call
                        logger.debug(f"[MAIN] Entered ignored tag: {tag!r}, output: {output!r}, buffer: {self.state['buffer']!r}")
                        return output
                # Could be a partial ignored tag
                for tag in self.ignore_tags:
                    if tag.startswith(chunk[i+1:]):
                        self.state['partial_tag_buffer'] = chunk[i:]
                        logger.debug(f"[MAIN] Buffering possible partial ignored tag: {self.state['partial_tag_buffer']!r}")
                        return output
                # Not an ignored tag, output up to and including this char
                if output:
                    logger.debug(f"[MAIN] Output before non-ignored tag: {output!r}")
                    return output
                else:
                    output += '<'
                    i += 1
                    continue
            elif c == '\n':
                output += '\n'
                i += 1
                continue
            else:
                output += c
                i += 1
        logger.debug(f"[MAIN] Final output: {output!r}")
        return output

@dataclass(kw_only=True)
class AIModel:
    model: t.Optional[str] = ''
    tiktoken_encoding: t.Optional[str] = "cl100k_base"
    support_functions: bool = False
    _provider_name: str = None
    stream_ignore_tags: t.List[str] = field(default_factory=list)
    _tag_parser: TagParser = field(init=False, default=None)

    def __post_init__(self):
        self._tag_parser = TagParser(self.stream_ignore_tags)

    def _should_stream_content(self, content: str) -> bool:
        """Determine if content should be streamed based on ignore tags"""
        return bool(self._tag_parser.text_to_stream_chunk(content))

    def text_to_stream_chunk(self, chunk: str) -> str:
        # If we have a buffered tag, prepend it to chunk
        if self.state['tag_buffer']:
            # If we find a newline, reset buffer and continue
            if '\n' in chunk:
                newline_pos = chunk.find('\n')
                self.state['tag_buffer'] = ''  # Reset buffer on newline
                return self.text_to_stream_chunk(chunk[newline_pos + 1:])
                
            chunk = self.state['tag_buffer'] + chunk
            self.state['tag_buffer'] = ''
        
        if not chunk:
            return ''

        # Split only on newlines to handle them separately
        for separator in ['\n', '\r\n']:
            if not separator in chunk:
                continue
            text_to_stream = []
            lines = chunk.split(separator)
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    line += separator
                processed = self._process_chunk(line)
                print(f'processed: {processed}, buffer: {self.state["tag_buffer"]}')
                if processed:
                    text_to_stream.append(processed)
            return ''.join(text_to_stream)

        processed = self._process_chunk(chunk)
        print(f'processed: {processed}, buffer: {self.state["tag_buffer"]}')
        return processed

    def _reset_tag_parser(self):
        """Reset tag parser state"""
        self._tag_parser.reset()

    @property
    def provider_name(self):
        return self.provider.value if not self.provider.is_custom() else self._provider_name

    @property
    def name(self) -> str:
        return "undefined_aimodel"

    def _decimal(self, value) -> Decimal:
        return Decimal(value).quantize(Decimal(".00001"))

    def get_params(self) -> t.Dict[str, t.Any]:
        return {}

    def get_metrics_data(self):
        return {}

    def call(
        self,
        current_messages: t.List[t.Dict[str, str]],
        max_tokens: t.Optional[int],
        tool_registry: t.Dict[str, ToolDefinition] = {},
        max_tool_iterations: int = 5,   # Safety limit for sequential calls
        stream_function: t.Callable = None,
        check_connection: t.Callable = None,
        stream_params: dict = {},
        client_secrets: dict = {},
        modelname='',
        prompt: 'Prompt' = None,
        context: str = '',
        test_data: dict = {},
        client: t.Any = None,
        **kwargs,
    ) -> AIResponse:
        """Common call implementation that handles streaming and tool calls."""
        # self._reset_tag_parser()  # Reset parser state for new call
        model_client = self.get_client(client_secrets)
        # Prepare streaming response
        stream_response = StreamingResponse(
            tool_registry=tool_registry,
            messages=current_messages
        )
        modelname = modelname.replace('/', '_').replace('-', '_')
        attempts = max_tool_iterations
        while attempts > 0:
            try:
                stream_response.update_to_another_attempt()
                stream_response = self.streaming(
                    client=model_client,
                    stream_response=stream_response,
                    max_tokens=max_tokens,
                    stream_function=stream_function,
                    check_connection=check_connection,
                    stream_params=stream_params,
                    **kwargs
                )
                logger.info(f'stream_response: {stream_response}')
                if stream_response.is_detected_tool_call:
                    parsed_tool_call = parse_tool_call_block(stream_response.content)

                    logger.info(f'parsed_tool_call {parsed_tool_call}')
                    if not parsed_tool_call or attempts <= 1:
                        stream_response.add_assistant_message()
                        self.save_call(stream_response, prompt, context, attempt=max_tool_iterations - attempts, client=client)
                        attempts -= 1
                        continue
                    # Execute tool call
                    self.handle_tool_call(parsed_tool_call, tool_registry)
                    # Add messages to history
                    logger.info(f'executed parsed_tool_call {parsed_tool_call}')
                    stream_response.add_tool_result(parsed_tool_call)
                    self.save_call(stream_response, prompt, context, attempt=max_tool_iterations - attempts, client=client)
                    attempts -= 1
                    continue
                stream_response.add_assistant_message()
                self.save_call(stream_response, prompt, context, test_data=test_data, client=client)
                logger.info(f'Passing execution {modelname}, finished. {attempts}')
                break
            except RetryableCustomError as e:
                logger.exception(f'RetryableCustomError {e}')
                attempts -= 1
                continue  
            except StopStreamingError as e:
                logger.exception(f'StopStreamingError {e}')
                stream_response.add_assistant_message()
                self.save_call(stream_response, prompt, context, attempt=max_tool_iterations - attempts, client=client)
                logger.info(f'Failing execution {modelname} w/ StopStreamingError, finished. {attempts}')
                raise e
        return stream_response


    def handle_tool_call(self, tool_call: ToolCallResult, tool_registry: t.Dict[str, ToolDefinition]) -> str:
        """Handle a tool call by executing the corresponding function from the registry."""
        function = tool_call.tool_name
        parameters = tool_call.parameters
        
        tool_function = tool_registry.get(function)
        if not tool_function:
            logger.warning(f"Tool '{function}' not found in registry")
            return json.dumps({"error": f"Tool '{function}' is not available."})
            
        try:
            logger.info(f"Executing tool '{function}' with parameters: {parameters}")
            result = tool_function.execution_function(**parameters)
            logger.info(f"Tool '{function}' executed successfully")
            tool_call.execution_result = result
            return json.dumps({"result": result})
        except StopStreamingError as e:
            logger.exception(f"Tool '{function}' execution stopped: {e}")
            tool_call.execution_result = str(e)
            raise e
        except Exception as e:
            result = f"Error executing tool '{function}', Please try second time."
            logger.exception(result, exc_info=e)
            tool_call.execution_result = result
            return json.dumps({"error": f"{result}: {str(e)}"})

    def streaming(
        self,
        client: t.Any,
        stream_response: StreamingResponse,
        max_tokens: int,
        stream_function: t.Callable,
        check_connection: t.Callable,
        stream_params: dict,
        **kwargs
    ) -> StreamingResponse:
        """Process streaming response. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement streaming method")

    def get_client(self, client_secrets: dict = {}) -> t.Any:
        """Get the client instance. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_client method")
    

    def calculate_budget_for_text(self, text: str) -> int:
        if not text:
            return 0
        return len(encoding.encode(text))
    
    def save_call(self, stream_response: StreamingResponse, prompt: "Prompt", context: str, attempt: int=0, test_data: dict = {}, client: t.Any = None):

        sample_budget = self.calculate_budget_for_text(
            stream_response.get_message_str()
        )
        stream_response.metrics.sample_tokens_used = sample_budget
        stream_response.metrics.prompt_tokens_used = self.calculate_budget_for_text(
            json.dumps(stream_response.messages)
        )
        stream_response.metrics.ai_model_details = (
            self.get_metrics_data()
        )
        stream_response.metrics.latency = current_timestamp_ms() - stream_response.started_tmst

        if settings.USE_API_SERVICE and client.api_token:
            stream_response.id = f"{prompt.id}#{stream_response.started_tmst}" + (f"#{attempt}" if attempt else "")
            client.worker.add_task(
                client.api_token,
                prompt.service_dump(),
                context,
                stream_response,
                {**test_data, "call_model": self.model}
            )
