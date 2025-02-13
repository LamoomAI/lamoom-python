import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock

from lamoom.ai_models.gemini.gemini_model import GeminiAIModel, FamilyModel
from lamoom.exceptions import RetryableCustomError, ConnectionLostError
from lamoom.responses import AIResponse, Prompt
from openai.types.chat import ChatCompletionMessage as Message


def test_gemini_ai_model_initialization():
    model_name = "gemini-1.5-flash"
    model = GeminiAIModel(model=model_name)

    assert model.model == model_name
    assert model.family == FamilyModel.flash.value


def test_gemini_ai_model_initialization_unknown_family():
    model_name = "gemini-unknown"
    model = GeminiAIModel(model=model_name)

    assert model.model == model_name
    assert model.family == FamilyModel.flash.value  # Default value as set in the class


@patch("lamoom.ai_models.gemini.gemini_model.genai.GenerativeModel")
def test_gemini_ai_model_call(mock_gen_model):
    mock_response = MagicMock()
    mock_response.text = "Test response"
    mock_gen_model().generate_content.return_value = mock_response

    model_name = "gemini-1.5-pro"
    model = GeminiAIModel(model=model_name)

    messages = [{"content": "Hello", "role": "user"}]
    max_tokens = 100
    client_secrets = {"api_key": "test_api_key"}

    response = model.call(messages, max_tokens, client_secrets)

    assert isinstance(response, AIResponse)
    assert response.message.content == "Test response"


@patch("lamoom.ai_models.gemini.gemini_model.genai.GenerativeModel")
def test_gemini_ai_model_call_with_stream(mock_gen_model):
    mock_chunk = MagicMock()
    mock_chunk.text = "chunk text"
    mock_gen_model().generate_content.return_value = iter([mock_chunk, mock_chunk])

    model_name = "gemini-1.5-pro"
    model = GeminiAIModel(model=model_name)

    messages = [{"content": "Hello", "role": "user"}]
    max_tokens = 100
    client_secrets = {"api_key": "test_api_key"}

    mock_check_connection = MagicMock(return_value=True)
    mock_stream_function = MagicMock()
    stream_params = {}

    response = model.call(
        messages,
        max_tokens,
        client_secrets,
        stream=True,
        stream_function=mock_stream_function,
        check_connection=mock_check_connection,
        stream_params=stream_params,
    )

    assert isinstance(response, AIResponse)
    assert "chunk text" in response.message.content
    mock_stream_function.assert_called()


def test_gemini_ai_model_get_prompt_price():
    model_name = "gemini-1.5-pro"
    model = GeminiAIModel(model=model_name)
    count_tokens = 500

    price = model.get_prompt_price(count_tokens)

    expected_price = Decimal(0.00175).quantize(Decimal("0.00001"))
    assert price == expected_price


def test_gemini_ai_model_get_prompt_price_500k_tokens():
    model_name = "gemini-1.5-pro"
    model = GeminiAIModel(model=model_name)
    count_tokens = 500_000

    price = model.get_prompt_price(count_tokens)

    expected_price = Decimal(3.5).quantize(Decimal("0.00001"))
    assert price == expected_price


def test_gemini_ai_model_get_sample_price():
    model_name = "gemini-1.5-pro"
    model = GeminiAIModel(model=model_name)
    budget_prompt = 100_000
    count_tokens = 500

    price = model.get_sample_price(budget_prompt, count_tokens)

    expected_price = Decimal(0.00525).quantize(Decimal("0.00001"))
    assert price == expected_price


@patch("lamoom.ai_models.gemini.gemini_model.genai.GenerativeModel")
def test_gemini_ai_model_call_with_connection_lost(mock_gen_model):
    mock_chunk = MagicMock()
    mock_chunk.text = "chunk text"
    mock_gen_model().generate_content.return_value = iter([mock_chunk] * 15)

    model_name = "gemini-1.5-pro"
    model = GeminiAIModel(model=model_name)

    messages = [{"content": "Hello", "role": "user"}]
    max_tokens = 100
    client_secrets = {"api_key": "test_api_key"}

    mock_check_connection = MagicMock(side_effect=[True, False])  # Simulate connection loss
    mock_stream_function = MagicMock()
    stream_params = {}

    with pytest.raises(RetryableCustomError):
        model.call(
            messages,
            max_tokens,
            client_secrets,
            stream=True,
            stream_function=mock_stream_function,
            check_connection=mock_check_connection,
            stream_params=stream_params,
        )


@patch("lamoom.ai_models.gemini.gemini_model.genai.GenerativeModel")
def test_gemini_ai_model_call_with_retryable_error(mock_gen_model):
    mock_gen_model().generate_content.side_effect = Exception("Test Exception")

    model_name = "gemini-1.5-pro"
    model = GeminiAIModel(model=model_name)

    messages = [{"content": "Hello", "role": "user"}]
    max_tokens = 100
    client_secrets = {"api_key": "test_api_key"}

    with pytest.raises(RetryableCustomError):
        model.call(messages, max_tokens, client_secrets)