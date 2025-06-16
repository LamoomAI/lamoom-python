from lamoom.ai_models.ai_model import AI_MODELS_PROVIDER, AIModel
import logging
import typing as t
from dataclasses import dataclass

from lamoom.ai_models.claude.constants import HAIKU, SONNET, OPUS
from lamoom.ai_models.constants import C_4K
from lamoom.responses import FINISH_REASON_ERROR, FINISH_REASON_FINISH, StreamingResponse
from lamoom.ai_models.tools.base_tool import TOOL_CALL_END_TAG, TOOL_CALL_START_TAG
from enum import Enum

from lamoom.exceptions import RetryableCustomError, ConnectionLostError
import anthropic

logger = logging.getLogger(__name__)


class FamilyModel(Enum):
    haiku = "Claude 3 Haiku"
    sonnet = "Claude 3 Sonnet"
    opus = "Claude 3 Opus"
    

@dataclass(kw_only=True)
class ClaudeAIModel(AIModel):
    max_tokens: int = C_4K
    api_key: str = None
    provider: AI_MODELS_PROVIDER = AI_MODELS_PROVIDER.CLAUDE
    family: str = None

    def __post_init__(self):
        if HAIKU in self.model:
            self.family = FamilyModel.haiku.value
        elif SONNET in self.model:
            self.family = FamilyModel.sonnet.value
        elif OPUS in self.model:
            self.family = FamilyModel.opus.value
        else:
            logger.info(
                f"Unknown family for {self.model}. Please add it obviously. Setting as Claude 3 Opus"
            )
            self.family = FamilyModel.opus.value
        logger.debug(f"Initialized ClaudeAIModel: {self}")

    def get_client(self, client_secrets: dict) -> anthropic.Anthropic:
        return anthropic.Anthropic(api_key=client_secrets.get("api_key"))

    def unify_messages_with_same_role(self, messages: t.List[dict]) -> t.List[dict]:
        result = []
        last_role = None
        for message in messages:
            if last_role != message.get("role"):
                result.append(message)
                last_role = message.get("role")
            else:
                result[-1]["content"] += message.get("content")
        return result

    def streaming(
        self,
        client: anthropic.Anthropic,
        stream_response: StreamingResponse,
        max_tokens: int,
        stream_function: t.Callable,
        check_connection: t.Callable,
        stream_params: dict,
        **kwargs
    ) -> StreamingResponse:
        """Process streaming response from Claude."""
        tool_call_started = False
        content = ""
        
        try:
            unified_messages = self.unify_messages_with_same_role(stream_response.messages)
            call_kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": unified_messages,
                **kwargs
            }
            # Extract system prompt if present
            system_prompt = []
            for i, msg in enumerate(unified_messages):
                if msg.get('role') == "system":
                    system_prompt.append(unified_messages.pop(i- len(system_prompt)).get('content'))
            
            if system_prompt:
                call_kwargs["system"] = '\n'.join(system_prompt)
            
            with client.messages.stream(**call_kwargs) as stream:
                for text_chunk in stream.text_stream:
                    if check_connection and not check_connection(**stream_params):
                        raise ConnectionLostError("Connection was lost!")
                    
                    stream_response.set_streaming()
                    content += text_chunk

                    # Only stream content if not in ignored tag
                    if stream_function or self._tag_parser.is_custom_tags():
                        text_chunk = self.text_to_stream_chunk(text_chunk)
                        if text_chunk and not tool_call_started:
                            stream_response.streaming_content += text_chunk
                            if stream_function:
                                stream_function(text_chunk, **stream_params)

                    # Check for tool call markers
                    if tool_call_started and TOOL_CALL_END_TAG in content:
                        stream_response.is_detected_tool_call = True
                        stream_response.content = content
                        logger.info(f'Found tool call request in {content}')
                        break
                    if TOOL_CALL_START_TAG in content:
                        if not tool_call_started:
                            tool_call_started = True
                        continue

            if stream_function or self._tag_parser.is_custom_tags():
                text_to_stream = self.text_to_stream_chunk('')
                if text_to_stream:
                    if stream_function:
                        stream_function(text_to_stream, **stream_params)
                    stream_response.streaming_content += text_to_stream
            
            stream_response.content = content
            stream_response.set_finish_reason(FINISH_REASON_FINISH)
            return stream_response
            
        except Exception as e:
            stream_response.content = content
            stream_response.set_finish_reason(FINISH_REASON_ERROR)
            logger.exception("Exception during stream processing", exc_info=e)
            raise RetryableCustomError(f"Claude AI stream processing failed: {e}") from e

    @property
    def name(self) -> str:
        return f"Claude {self.family}"

    def get_params(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
        }

    def get_metrics_data(self) -> t.Dict[str, t.Any]:
        return {
            "model": self.model,
            "family": self.family,
            "max_tokens": self.max_tokens,
        }
