import json
import re
import typing as t
from dataclasses import dataclass, field
from enum import Enum
import logging
from _decimal import Decimal
from lamoom.prompt.base_prompt import BasePrompt
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

    def __init__(self, ignore_tags: t.List[str] = None, writing_tags: t.List[str] = None):
        self.ignore_tags = list(set(ignore_tags or []))
        self.writing_tags = list(set(writing_tags or []))
        self.reset()
    
    def is_custom_tags(self):
        return bool(self.ignore_tags or self.writing_tags)

    def reset(self):
        self.state = {
            'buffer': '',
            'ignored_tag': None,
            'writing_tag': None,
            'tags': [],
            'in_ignored_tag': False,
            'in_writing_tag': True,
        }
        if self.writing_tags:
            self.state['in_writing_tag'] = False
        if self.ignore_tags:
            self.state['in_ignored_tag'] = False

    def parse_tags(self, chunk: str) -> t.List[str]:
        matches = []
        ignored_length = len(self.ignore_tags)
        for i, tag in enumerate(self.ignore_tags + self.writing_tags):
            is_ignored = i < ignored_length
            opening_tag = f'<{tag}(>| )'
            closing_tag = f'</{tag}>'
            for match in re.finditer(opening_tag, chunk):
                matched_tag = {
                    'tag': tag, 'type': 'opening', 
                    'position': match.start(), 
                    'is_ignored': is_ignored,
                    'end_position': match.end()
                }
                matches.append(matched_tag)
                logger.debug(f"[PARSE_TAGS] match {opening_tag}: {matched_tag}")
            for match in re.finditer(closing_tag, chunk):
                matched_tag = {
                    'tag': tag, 'type': 'closing', 
                    'position': match.start(), 
                    'is_ignored': is_ignored,
                    'end_position': match.end()
                    }
                matches.append(matched_tag)
                logger.debug(f"[PARSE_TAGS] closing match: {matched_tag}")
        matches.sort(key=lambda x: x['position'])
        matches.append({'tag': None, 'type': 'end', 'position': len(chunk), 
                            'is_ignored': False, 'end_position': len(chunk)})
        return matches

    def text_to_stream_chunk(self, chunk: str) -> str:
        # Split only on newlines to handle them separately
        for separator in ['<']:
            if not separator in chunk:
                continue
            text_to_stream = []
            lines = chunk.split(separator)
            for i, line in enumerate(lines):
                if not line and i == 0:
                    continue
                if i != 0:
                    line = separator + line
                processed = self._text_to_stream_chunk(line)
                if processed:
                    text_to_stream.append(processed)
            return ''.join(text_to_stream)

        processed = self._text_to_stream_chunk(chunk)
        return processed


    def _text_to_stream_chunk(self, incoming_chunk: str) -> str:
        logger = logging.getLogger("TagParser")
        logger.debug(f"[INPUT] chunk: {incoming_chunk!r}: {self.state['buffer']}, tags: {self.state['tags']}")

        # Always process buffer first if it exists
        chunk = self.state['buffer'] + incoming_chunk
        self.state['buffer'] = ''
        if '<' in chunk and not '>' in chunk and (chunk.rfind('<') - len(chunk)) < 5 and incoming_chunk:
            # wait to get more data
            self.state['buffer'] = chunk
            logger.debug(f"[STREAM] waiting for more data: {chunk}")
            return ''
        if '<' not in chunk:
            logger.debug(f"[STREAM] not '<' in chunk: {chunk}, in_ignored_tag: {self.state['in_ignored_tag']}, in_writing_tag: {self.state['in_writing_tag']}")
            if not self.state['in_ignored_tag'] and self.state['in_writing_tag']:
                return chunk
        elif '>' not in chunk and incoming_chunk and len(chunk) < self.MAX_BUFFER_SIZE:
            self.state['buffer'] = chunk
            return ''

        if len(chunk) > self.MAX_BUFFER_SIZE:
            # always output what wasn't added in FIFO buffer if it's not in ignored_tag;
            chunk_to_process = chunk[:len(chunk) - self.MAX_BUFFER_SIZE]
            if not self.state['in_ignored_tag'] and self.state['in_writing_tag']:
                chunk = chunk[len(chunk) - self.MAX_BUFFER_SIZE:]
                return chunk_to_process

        tag_matches = self.parse_tags(chunk)
        last_match_index = 0
        # not_ignored, ignored
        for match in tag_matches:
            value = chunk[last_match_index:match['position']]
            value_till_end = chunk[last_match_index:match['end_position']]
            last_match_index = match['end_position']
            is_ignored = self.state['in_ignored_tag']
            in_writing_tag = self.state['in_writing_tag']

            logger.debug(f"[STREAM] match {match['type']} {match['tag']}: {value}, ignored: {is_ignored}, writing: {in_writing_tag}")
            if match['type'] == 'opening':
                self.state['tags'].append(match['tag'])
                if match['is_ignored']:
                    self.state['ignored_tag'] = match['tag']
                    self.state['in_ignored_tag'] = True
                else:
                    # TODO add writing tag closing like '>' which can be in inf characters
                    self.state['writing_tag'] = match['tag']
                    self.state['in_writing_tag'] = True
                
                if not value:
                    continue
                if not is_ignored and in_writing_tag:
                    logger.debug(f"[STREAM] Writing match: {match}: {value}: {self.state}")
                    self.state['buffer'] = chunk[match['end_position']:]
                    return value
                logger.debug(f"[STREAM][NOT WRITING] {match}: {value}: {self.state}")
                
            if match['type'] == 'closing':
                try:
                    index_of_closing_tag = len(self.state['tags']) - self.state['tags'][::-1].index(match['tag']) - 1
                except ValueError:
                    index_of_closing_tag = len(self.state['tags'])
                self.state['buffer'] = chunk[match['end_position']:]
                logger.debug(f"[STREAM] Closing match: {match}: {value} index_of_opening_tag: {index_of_closing_tag}, buffer {self.state['buffer']}")
                # if it was not opened
                if index_of_closing_tag == len(self.state['tags']):
                    logger.debug(f"[STREAM] Closing tag not opened: {match}: {value}")
                    if in_writing_tag and not is_ignored:
                        return value_till_end
                    return ''
                pop_all_tags_starting_from_closed_tag = self.state['tags'][index_of_closing_tag:]
                self.state['tags'] = self.state['tags'][:index_of_closing_tag]
                logger.debug(f"[STREAM] self.state['tags']: {self.state['tags']}")
                # what should we do with chunk?
                partial_chunk = value
                if match['is_ignored']:
                    list_tags = [tag for tag in self.state['tags'] if tag in self.ignore_tags]
                    self.state['in_ignored_tag'] = bool(list_tags)
                    self.state['ignored_tag'] = list_tags[-1] if list_tags else None
                else:
                    list_tags = [tag for tag in self.state['tags'] if tag in self.writing_tags]
                    self.state['in_writing_tag'] = bool(list_tags)
                    self.state['writing_tag'] = list_tags[-1] if list_tags else None
                logger.debug(f"[STREAM] in_ignored { self.state['in_ignored_tag']}, in_writing_tag: {self.state['in_writing_tag']}")
                if not partial_chunk:
                    logger.debug(f"[STREAM] closing tag. partial_chunk: {partial_chunk}: {self.state}. Continueing")
                    continue
                if not is_ignored and in_writing_tag:
                    return partial_chunk
            
            if match['type'] == 'end':
                self.state['buffer'] = ''
                if in_writing_tag and not is_ignored:
                    logger.debug(f"[STREAM] End match: {match}: {value}")
                    return value 
        self.state['buffer'] = chunk[last_match_index:]
        return ''

@dataclass(kw_only=True)
class AIModel:
    model: t.Optional[str] = ''
    tiktoken_encoding: t.Optional[str] = "cl100k_base"
    support_functions: bool = False
    _provider_name: str = None
    _tag_parser: TagParser = field(default=None, init=False)

    def _init_tag_parser(self, ignore_tags: t.List[str] = None, writing_tags: t.List[str] = None):
        self._tag_parser = TagParser(ignore_tags=ignore_tags, writing_tags=writing_tags)

    def text_to_stream_chunk(self, chunk: str) -> str:
        """Process incoming chunk of text and return the parsed result."""
        if not self._tag_parser:
            return chunk
        return self._tag_parser.text_to_stream_chunk(chunk)

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
        ignore_tags: t.List[str] = None,
        writing_tags: t.List[str] = None,
        **kwargs,
    ) -> AIResponse:
        """Common call implementation that handles streaming and tool calls."""
        # self._reset_tag_parser()  # Reset parser state for new call
        model_client = self.get_client(client_secrets)
        # Prepare streaming response
        stream_response = StreamingResponse(
            tool_registry=tool_registry,
            messages=current_messages,
            prompt=BasePrompt(
                messages=current_messages,
                functions=kwargs.get("tools"),
                max_tokens=max_tokens,
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
            )
        )
        modelname = modelname.replace('/', '_').replace('-', '_')
        attempts = max_tool_iterations
        self._init_tag_parser(ignore_tags=ignore_tags, writing_tags=writing_tags)
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
    
    def save_call(self, stream_response: StreamingResponse, prompt: "Prompt", context: dict, attempt: int=0, test_data: dict = {}, client: t.Any = None):

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
