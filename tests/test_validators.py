import pytest

from lamoom import JSONValidator, XMLValidator, YAMLValidator
from lamoom import AIResponse

format = {
    "validator_id": "test_validator",
    "validator_type": "json",
    "schema": {
        "type": "object",
        "required": ["statements", "questions", "name"],
        "properties": {
            "name": {"type": "string"},
            "statements": {
                "type": "array",
                "items": {"type": "string"}
            },
            "questions": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    },
    "retry": 3,
    "retry_rules": {
        "invalid_structure": 2,
        "missing_field": 2,
        "invalid_type": 1
    }
}

prompt_id = "Statements and Questions"
schema = format["schema"]
retry_count = format["retry"]
retry_rules = format["retry_rules"]

@pytest.fixture
def json_validator():
    validator = JSONValidator("test_json", schema, retry_count, retry_rules)
    validator.attach_to_promt(prompt_id)
    return validator

def test_valid_json(json_validator):
    response = AIResponse(_response="""```json
{
    "statements": ["Statement 1", "Statement 2"],
    "questions": ["Question 1", "Question 2"],
    "name": "Test Name",
    "extra_field": "Extra value"
}
```
""")
    json_validator.validate(response)
    assert not json_validator.has_errors()

def test_invalid_json(json_validator):
    response = AIResponse(_response="""```json
{
    name: "Test Name",
    "statements": ["Statement 1"],
    "questions": ["Question 1"]
}
```
""")
    json_validator.validate(response)
    assert len(json_validator.get_errors()) == 1
    assert json_validator.get_errors()[0]['type'] == "invalid_structure"

def test_missing_required_field_json(json_validator):
    response = AIResponse(_response="""```json
{
    "statements": ["Statement 1"],
    "questions": ["Question 1"]
}
```
""")
    json_validator.validate(response)
    assert len(json_validator.get_errors()) == 1
    assert json_validator.get_errors()[0]['type'] == "missing_field"

def test_invalid_type_json(json_validator):
    response = AIResponse(_response="""```json
{
    "name": "Test Name",
    "statements": ["Statement 1"],
    "questions": "This should be an array"
}
```
""")
    json_validator.validate(response)
    assert len(json_validator.get_errors()) == 1
    assert json_validator.get_errors()[0]['type'] == "invalid_type"  

def test_no_json_content(json_validator):
    response = AIResponse(_response="""No JSON content here!""")
    json_validator.validate(response)
    assert len(json_validator.get_errors()) == 1
    assert json_validator.get_errors()[0]['type'] == "invalid_structure" 

@pytest.fixture
def xml_validator():
    validator = XMLValidator("test_xml", schema, retry_count, retry_rules)
    validator.attach_to_promt(prompt_id)
    return validator

def test_valid_xml(xml_validator):
    response = AIResponse(_response="""
```xml
<root>
    <statements>
        <statement>Statement 1</statement>
        <statement>Statement 2</statement>
    </statements>
    <questions>
        <question>Question 1 </question>
        <question>Question 2 </question>                  
    </questions>
    <name> Test Name </name>
</root>
```              
""")
    xml_validator.validate(response)
    print(xml_validator.get_errors())
    assert not xml_validator.has_errors()

def test_invalid_xml(xml_validator):
    response = AIResponse(_response="""```xml
<root>
    <name>Test Name</name>
    <statements>
    <item>Statement 1
    </statements>
</root>
```
""")  
    xml_validator.validate(response)
    assert len(xml_validator.get_errors()) == 1
    assert xml_validator.get_errors()[0]['type'] == "invalid_structure"

def test_missing_required_field_xml(xml_validator):
    response = AIResponse(_response="""
```xml
<root>
    <statements>
        <statement>Statement 1</statement>
        <statement>Statement 2</statement>
    </statements>
    <questions>
        <question>Question 1 </question>
        <question>Question 2 </question>                  
    </questions>
</root>
```
""") 
    xml_validator.validate(response)
    assert len(xml_validator.get_errors()) == 1
    assert xml_validator.get_errors()[0]['type'] == "missing_field"

def test_invalid_type_xml(xml_validator):
    response = AIResponse(_response="""```xml
<root>
    <name>Test Name</name>
    <statements>
        <item>Statement 1</item>
    </statements>
    <questions>This should be array</questions>
</root>
```
""")
    xml_validator.validate(response)
    assert len(xml_validator.get_errors()) == 1
    assert xml_validator.get_errors()[0]['type'] == "invalid_type"

def test_no_xml_content(xml_validator):
    response = AIResponse(_response="""No XML content here!""")
    xml_validator.validate(response)
    assert len(xml_validator.get_errors()) == 1
    assert xml_validator.get_errors()[0]['type'] == "invalid_structure"


@pytest.fixture
def yaml_validator():
    validator = YAMLValidator("test_yaml", schema, retry_count, retry_rules)
    validator.attach_to_promt(prompt_id)
    return validator

def test_valid_yaml(yaml_validator):
    response = AIResponse(_response="""
```yaml
statements:
  - Statement 1
  - Statement 2
questions:
  - Question 1
  - Question 2
name: Test Name
```
""")
    yaml_validator.validate(response)
    assert not yaml_validator.has_errors()

def test_invalid_yaml(yaml_validator):
    response = AIResponse(_response="""
```yaml
name: Test Name
statements:
Statement 1
questions:
Question 1
julia
```
""")
    yaml_validator.validate(response)
    assert len(yaml_validator.get_errors()) == 1
    assert yaml_validator.get_errors()[0]['type'] == "invalid_structure"

def test_missing_required_field_yaml(yaml_validator):
    response = AIResponse(_response="""
```yaml
statements:
  - Statement 1
questions:
  - Question 1
```
""")
    yaml_validator.validate(response)
    assert len(yaml_validator.get_errors()) == 1
    assert yaml_validator.get_errors()[0]['type'] == "missing_field"

def test_invalid_type_yaml(yaml_validator):
    response = AIResponse(_response="""
```yaml
name: Test Name
statements:
 - Statement 1
questions: This should be an array
```
""")
    yaml_validator.validate(response)
    assert len(yaml_validator.get_errors()) == 1
    assert yaml_validator.get_errors()[0]['type'] == "invalid_type"

def test_no_yaml_content(yaml_validator):
    response = AIResponse(_response="""No YAML content here!""")
    yaml_validator.validate(response)
    assert len(yaml_validator.get_errors()) == 1
    assert yaml_validator.get_errors()[0]['type'] == "invalid_structure"


@pytest.fixture
def empty_json_validator():
    validator = JSONValidator("test_empty_json")
    validator.attach_to_promt("test_prompt")
    return validator

def test_no_empty_validator_json_content(empty_json_validator):
    response = AIResponse(_response="""No JSON content here!""")
    empty_json_validator.validate(response)
    assert len(empty_json_validator.get_errors()) == 1
    assert empty_json_validator.get_errors()[0]['type'] == "invalid_structure"

def test_invalid_empty_validator_json(empty_json_validator):
    response = AIResponse(_response="""```json
{
    name: "Test Name",
    "statements": ["Statement 1"],
    "questions": ["Question 1"]
}
```
""")
    empty_json_validator.validate(response)
    assert len(empty_json_validator.get_errors()) == 1
    assert empty_json_validator.get_errors()[0]['type'] == "invalid_structure"

def test_valid_empty_validator_json(empty_json_validator):
    response = AIResponse(_response="""```json
{
    "statements": ["Statement 1", "Statement 2"],
    "questions": ["Question 1", "Question 2"],
    "name": "Test Name",
    "extra_field": "Extra value"
}
```
""")
    empty_json_validator.add_field_to_validate("statements", "array")
    empty_json_validator.validate(response)
    assert not empty_json_validator.has_errors()


def test_missing_required_field_empty_validator_json(empty_json_validator):
    response = AIResponse(_response="""```json
{
    "statements": ["Statement 1"],
    "questions": ["Question 1"]
}
```
""")
    empty_json_validator.add_field_to_validate("name", "string", minLength = 1)
    empty_json_validator.validate(response)
    assert len(empty_json_validator.get_errors()) == 1
    assert empty_json_validator.get_errors()[0]['type'] == "missing_field"

def test_invalid_type_empty_validator_json(empty_json_validator):
    response = AIResponse(_response="""```json
{
    "name": "Test Name",
    "statements": ["Statement 1"],
    "questions": "This should be an array"
}
```
""")
    empty_json_validator.add_field_to_validate("questions", "array")
    empty_json_validator.validate(response)
    assert len(empty_json_validator.get_errors()) == 1
    assert empty_json_validator.get_errors()[0]['type'] == "invalid_type"

field_type = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["title", "content"],
            "properties": {
            "title": {
                "type": "string",
                "minLength": 1
            },
            "content": {
                "type": "array" 
                }
            }
        }
    }

@pytest.fixture
def empty_extra_json_validator():
    validator = JSONValidator("test_empty_extra_json")
    validator.attach_to_promt("test_extra_prompt")
    return validator

def test_no_extra_empty_validator_json_content(empty_extra_json_validator):
    response = AIResponse(_response="""No JSON content here!""")
    empty_extra_json_validator.validate(response)
    assert len(empty_extra_json_validator.get_errors()) == 1
    assert empty_extra_json_validator.get_errors()[0]['type'] == "invalid_structure"

def test_invalid_extra_empty_validator_json(empty_extra_json_validator):
    response = AIResponse(_response="""
```json
{
    name: "Test Name",
    "statements": ["Statement 1"],
    "questions": ["Question 1"]
}
```
""")
    empty_extra_json_validator.validate(response)
    assert len(empty_extra_json_validator.get_errors()) == 1
    assert empty_extra_json_validator.get_errors()[0]['type'] == "invalid_structure"

def test_valid_extra_empty_validator_json(empty_extra_json_validator):
    response = AIResponse(_response="""
```json
{
    "statements": [{"title": "Statement1", "content": ["Some content"]}],
    "questions": ["Question 1", "Question 2"],
    "name": "Test Name",
    "extra_field": "Extra value"
}
```
""")
    empty_extra_json_validator.add_field_to_validate("statements", field_type)
    print(empty_extra_json_validator.schema)
    empty_extra_json_validator.validate(response)
    print(empty_extra_json_validator.get_errors())
    assert not empty_extra_json_validator.has_errors()

def test_missing_required_field__extra_empty_validator_json(empty_extra_json_validator):
    response = AIResponse(_response="""```json
{
    "statements": [{"title": "Statement1", "content": ["Some content"]}],
    "questions": ["Question 1", "Question 2"],
    "name": [{"name": "Test Name"}],
    "extra_field": "Extra value"
}
```
""")
    empty_extra_json_validator.add_field_to_validate(
        "name", 
        {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "subname"],
            },
        }
        )
    empty_extra_json_validator.validate(response)
    assert len(empty_extra_json_validator.get_errors()) == 1
    assert empty_extra_json_validator.get_errors()[0]['type'] == "missing_field"

def test_invalid_type_extra_empty_validator_json(empty_extra_json_validator):
    response = AIResponse(_response="""```json
{
    "statements": [{"title": "Statement1", "content": ["Some content"]}],
    "questions": ["Question 1", "Question 2"],
    "name": [{"name": "Test Name", "subname": "Test Subname"}],
    "extra_field": "Extra value"
}
```
""")
    empty_extra_json_validator.add_field_to_validate(
        "questions", 
        {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["question", "subquestion"],
            },
        }
        )
    empty_extra_json_validator.validate(response)
    assert len(empty_extra_json_validator.get_errors()) == 1
    assert empty_extra_json_validator.get_errors()[0]['type'] == "invalid_type"