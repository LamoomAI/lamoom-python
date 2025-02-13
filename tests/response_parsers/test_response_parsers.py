import pytest
from lamoom.responses import AIResponse
from lamoom.exceptions import NotParsedResponseException
from lamoom.response_parsers.response_parser import get_yaml_from_response, get_json_from_response, _get_format_from_response, Tag


def test_get_yaml_from_response_valid_yaml():
    response = AIResponse(_response="```yaml\nkey: value\n```")
    tagged_content = get_yaml_from_response(response)
    
    assert tagged_content.content == 'key: value'
    assert tagged_content.parsed_content == {'key': 'value'}
    assert tagged_content.start_ind == 7
    assert tagged_content.end_ind == 19

def test_get_yaml_from_response_invalid_yaml():
    response = AIResponse(_response="```yaml\nkey: value\n```")
    tagged_content = get_yaml_from_response(response)
    
    assert tagged_content.content == 'key: value'
    assert tagged_content.parsed_content == {"key": "value"}

def test_get_json_from_response_valid_json():
    response = AIResponse(_response="```json\n{\"key\": \"value\"}\n```")
    tagged_content = get_json_from_response(response)
    
    assert tagged_content.content == '{"key": "value"}'
    assert tagged_content.parsed_content == {"key": "value"}
    assert tagged_content.start_ind == 7
    assert tagged_content.end_ind == 24

def test_get_json_from_response_invalid_json():
    response = AIResponse(_response="```json\n{key: value}\n```")
    
    with pytest.raises(NotParsedResponseException):
        get_json_from_response(response)

def test__get_format_from_response():
    response = AIResponse(_response="```json\n{\"key\": \"value\"}\n```")
    tags = [Tag("```json", "```", 0)]
    content, start_ind, end_ind = _get_format_from_response(response, tags)
    
    assert content == '{"key": "value"}'

def test__get_format_from_response_no_tags():
    response = AIResponse(_response="No tags here")
    tags = [Tag("```json", "```", 0)]
    content, start_ind, end_ind = _get_format_from_response(response, tags)
    assert content is None

