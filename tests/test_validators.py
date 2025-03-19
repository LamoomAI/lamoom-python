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
shema = format["schema"]
retry_count = format["retry"]
retry_rules = format["retry_rules"]

@pytest.fixture
def json_validator():
    validator = JSONValidator(prompt_id, "test_json", shema, retry_count, retry_rules)
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
    validator = XMLValidator(prompt_id, "test_xml", shema, retry_count, retry_rules)
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
    validator = YAMLValidator(prompt_id, "test_yaml", shema, retry_count, retry_rules)
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