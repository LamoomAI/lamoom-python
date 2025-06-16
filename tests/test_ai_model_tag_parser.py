import pytest
from unittest.mock import Mock, patch
from lamoom.ai_models.ai_model import AIModel, TagParser
from lamoom.responses import StreamingResponse
from lamoom.ai_models.tools.base_tool import TOOL_CALL_START_TAG, TOOL_CALL_END_TAG


class TestTagParser:
    @pytest.fixture
    def parser(self):
        return TagParser(ignore_tags=['think', 'reason'])

    def test_basic_text(self, parser):
        # Test text without tags
        assert parser.text_to_stream_chunk("Hello world") == "Hello world"
        assert parser.text_to_stream_chunk("") == ""

    def test_ignored_tags(self, parser):
        parser.reset()
        # Test ignored tags
        assert parser.text_to_stream_chunk("Hello <think>world" ) == "Hello "
        assert parser.text_to_stream_chunk("</think>") == ""
        assert parser.text_to_stream_chunk("Hello <reason>world</") == "Hello "
        # still in ignored tag reason
        assert parser.text_to_stream_chunk("Hello <think>world</think> there") == ""
        assert parser.text_to_stream_chunk("</") == ""
        assert parser.text_to_stream_chunk("reason>") == ""  # Close ignored tag
        assert parser.text_to_stream_chunk(" there") == " there"

    def test_non_ignored_tags(self, parser):
        parser.reset()
        # Test non-ignored tags
        assert parser.text_to_stream_chunk("Hello <other>world</other>") == "Hello <other>world</other>"
        assert parser.text_to_stream_chunk("Hello <think>world</other>") == "Hello "
        print(parser.state)
        assert parser.text_to_stream_chunk("") == ''

    def test_partial_tags(self, parser):
        parser.reset()
        # Test partial tag handling
        assert parser.text_to_stream_chunk("<t") == ""  # Buffer partial ignored tag
        assert parser.text_to_stream_chunk("hink>") == ""  # Complete ignored tag
        assert parser.text_to_stream_chunk("content") == ""  # Content in ignored tag
        assert parser.text_to_stream_chunk("</think>") == ""  # Close ignored tag
        assert parser.text_to_stream_chunk("after") == "after"  # Content after ignored tag

        # Test partial non-ignored tag
        assert parser.text_to_stream_chunk("hey ") == "hey "  # Output partial non-ignored tag
        assert parser.text_to_stream_chunk("<") == ""  # Output partial non-ignored tag
        assert parser.text_to_stream_chunk("o") == ""  # Output partial non-ignored tag
        assert parser.text_to_stream_chunk("ther>") == "<other>"  # Complete non-ignored tag

    def test_writing_tags(self, parser):
        parser = TagParser(writing_tags=['write'])
        # Test partial tag handling
        assert parser.text_to_stream_chunk("Simple") == ""
        assert parser.text_to_stream_chunk("<w") == ""
        assert parser.text_to_stream_chunk("rite>") == ""  # Complete ignored tag
        assert parser.text_to_stream_chunk("now") == "now"  # Complete ignored tag
        assert parser.text_to_stream_chunk(" ") == " "  # Content in ignored tag
        assert parser.text_to_stream_chunk("<") == ""  # Close ignored tag
        assert parser.text_to_stream_chunk("/") == ""
        assert parser.text_to_stream_chunk("write>") == ""  # Close ignored tag
        assert parser.text_to_stream_chunk("after") == ""  # Content after ignored tag

        # Test partial non-ignored tag
        assert parser.text_to_stream_chunk("hey ") == ""  # Output partial non-ignored tag
        assert parser.text_to_stream_chunk("<") == ""  # Output partial non-ignored tag
        assert parser.text_to_stream_chunk("o") == ""  # Output partial non-ignored tag
        assert parser.text_to_stream_chunk("ther>") == ""  # Complete non-ignored tag


class TestAIModelTagStreaming:
    @pytest.fixture
    def model(self):
        model = AIModel()
        model._init_tag_parser(ignore_tags=['think', 'reason'], writing_tags=[])
        return model

    @pytest.fixture
    def mock_stream_function(self):
        return Mock()

    @pytest.fixture
    def stream_response(self):
        return StreamingResponse(messages=[], tool_registry={})

    def test_basic_streaming(self, model, mock_stream_function, stream_response):
        # Test basic streaming with ignored tags
        chunks = [
            "Hello ",
            "<think>",
            "ignored",
            "</think>",
            " world",
            ""
        ]
        
        for chunk in chunks:
            stream_chunk = model.text_to_stream_chunk(chunk)
            if not stream_chunk:
                continue
            mock_stream_function(stream_chunk)
        
        # Only non-ignored content should be streamed
        assert mock_stream_function.call_count == 2
        mock_stream_function.assert_any_call("Hello ")
        mock_stream_function.assert_any_call(" world")

    def test_tool_calls_with_tags(self, model, mock_stream_function, stream_response):
        # Test tool calls with ignored tags
        content = f"Before {TOOL_CALL_START_TAG}<think>ignored</think>{TOOL_CALL_END_TAG} After"
        
        # Simulate streaming
        for chunk in content:
            stream_chunk = model.text_to_stream_chunk(chunk)
            if not stream_chunk:
                continue
            mock_stream_function(stream_chunk)
        
        # Tool call content should be accumulated but not streamed if in ignored tag
        assert mock_stream_function.call_count > 0
        assert "Before" in "".join(call[0][0] for call in mock_stream_function.call_args_list)
        assert "After" in "".join(call[0][0] for call in mock_stream_function.call_args_list)

    def test_nested_streaming(self, model, mock_stream_function, stream_response):
        # Test nested tag streaming
        chunks = [
            "Start ",
            "<think>",
            "outer ",
            "<code>",
            "inner",
            "</code>",
            " more",
            "</think>",
            " End",
            ""
        ]
        
        for chunk in chunks:
            stream_chunk = model.text_to_stream_chunk(chunk)
            if not stream_chunk:
                continue
            mock_stream_function(stream_chunk)
        
        # Only content outside ignored tags should be streamed
        assert mock_stream_function.call_count == 2
        mock_stream_function.assert_any_call("Start ")
        mock_stream_function.assert_any_call(" End")

    def test_partial_tag_streaming(self, model, mock_stream_function, stream_response):
        # Test streaming with partial tags
        chunks = [
            "Start ",
            "<t",
            "hink>",
            "ignored",
            "</t",
            "hink>",
            " End",
            ""
        ]
        
        for chunk in chunks:
            stream_chunk = model.text_to_stream_chunk(chunk)
            if not stream_chunk:
                continue
            mock_stream_function(stream_chunk)
        
        print(mock_stream_function.call_args_list)
        # Partial tags should be handled correctly
        assert mock_stream_function.call_count == 2
        mock_stream_function.assert_any_call("Start ")
        mock_stream_function.assert_any_call(" End")

    def test_newline_streaming(self, model, mock_stream_function, stream_response):
        # Test streaming with newlines
        chunks = [
            "Line 1<<",
            "<t",
            "hink>ignored</t",
            "hink>",
            "Line 2<<",
            ""
        ]
        
        for chunk in chunks:
            stream_chunk = model.text_to_stream_chunk(chunk)
            if not stream_chunk:
                continue
            mock_stream_function(stream_chunk)
    
        print(mock_stream_function.call_args_list)
        # Newlines should break tags and be streamed
        assert mock_stream_function.call_count == 4
        assert "Line 1<<" in "".join(call[0][0] for call in mock_stream_function.call_args_list)
        assert "Line 2<<" in "".join(call[0][0] for call in mock_stream_function.call_args_list)
