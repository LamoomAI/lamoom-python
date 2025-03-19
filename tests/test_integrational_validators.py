import dotenv
import os
from lamoom import Lamoom
from lamoom import JSONValidator, XMLValidator, YAMLValidator
from lamoom import Prompt

from pytest import fixture

json_agent = Prompt("Roman Empire json")
json_agent.add("""
Generate 10 statements about the Roman Empire and corresponding questions.
Use the following JSON format for your answer:

```json
{
    "statements": [],
    "questions": [],
    "name": ""
}
```
""", role="system")

xml_agent = Prompt("Roman Empire xml")
xml_agent.add("""
Generate 10 statements about the Roman Empire and corresponding questions.
Use the following XML format for your answer:

```xml
<root>
    <statements>
        <statement></statement>
        <statement></statement> 
    </statements>
    <questions>
        <question></question>
        ...
    </questions>
    <name></name>
</root>
```              
""", role="system")

yaml_agent = Prompt("Roman Empire yaml")
yaml_agent.add("""
Generate 10 statements about the Roman Empire and corresponding questions.
Use the following YAML format for your answer:

```yaml
statements:
  - ""
  - ""
  ...
questions:
  - ""
  - ""
  ...
name: ""
```               
""", role="system")


extra_json_agent = Prompt("Roman Empire json2")
extra_json_agent.add("""
Create a JSON object that conforms to the following schema:
#it's important - ```json end in the end ```
```json 
{
  "type": "object",
  "required": ["title", "author", "chapters"],
  "properties": {
    "title": {
      "type": "string",
      "minLength": 1
    },
    "author": {
      "type": "object",
      "required": ["name", "birthDate"],
      "properties": {
        "name": {
          "type": "string",
          "minLength": 1
        },
        "birthDate": {
          "type": "string",
          "format": "date"
        },
        "nationality": {
          "type": "string"
        }
      }
    },
    "chapters": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["title", "pageCount", "sections"],
        "properties": {
          "title": {
            "type": "string",
            "minLength": 1
          },
          "pageCount": {
            "type": "integer",
            "minimum": 1
          },
          "sections": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["heading", "content"],
              "properties": {
                "heading": {
                  "type": "string",
                  "minLength": 1
                },
                "content": { 
                  "type": "string"
                },
                "subSections": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "required": ["subHeading", "text"],
                    "properties": {
                      "subHeading": {
                        "type": "string",
                        "minLength": 1
                      },
                      "text": {
                        "type": "string"
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    },
    "published": {
      "type": "boolean"
    },
    "publicationDate": {
      "type": "string",
      "format": "date"
    }
  }
}
```
                
Please generate a sample data object that matches this schema and includes:
- A book title
- Author information (name, birth date, nationality)
- A list of chapters with titles, page counts, and sections, including sub-sections
- A publication flag and publication date
""", role="system")

two_json_agent = Prompt("Roman Empire two json")
two_json_agent.add("""
Generate 10 statements about the Roman Empire and corresponding questions.
Use the following JSON format for your answer:
                   
```json
{
    "statements": [],
    "name": ""
}
```
                   
```json
{
    "questions": [],
    "name": ""
}
```

```json
{
    "statements": [],
    "questions": [],
    "name": ""
}
```              
""", role="system")


base_format = {
  "validator_id": "base_test",
  "validator_type": "json",
  "schema": {
    "type": "object",
    "required": ["statements", "questions", "name"],
    "properties": {
            "name": { "type": "string" },
            "statements": {
                "type": "array",
                "items": { "type": "string" }
            },
            "questions": {
                "type": "array",
                "items": { "type": "string" }
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

extra_format = {
  "validator_id": "extra_test",  
  "validator_type": "json",
  "schema": {
    "type": "object",
    "required": ["title", "author", "chapters"],
    "properties": {
      "title": {
        "type": "string",
        "minLength": 1
      },
      "author": {
        "type": "object",
        "required": ["name", "birthDate"],
        "properties": {
          "name": {
            "type": "string",
            "minLength": 1
          },
          "birthDate": {
            "type": "string",
            "format": "date"
          },
          "nationality": {
            "type": "string"
          }
        }
      },
      "chapters": {
        "type": "array",
        "items": {
          "type": "object",
          "required": ["title", "pageCount", "sections"],
          "properties": {
            "title": {
              "type": "string",
              "minLength": 1
            },
            "pageCount": {
              "type": "integer",
              "minimum": 1
            },
            "sections": {
              "type": "array",
              "items": {
                "type": "object",
                "required": ["heading", "content"],
                "properties": {
                  "heading": {
                    "type": "string",
                    "minLength": 1
                  },
                  "content": {
                    "type": "string"
                  },
                  "subSections": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "required": ["subHeading", "text"],
                      "properties": {
                        "subHeading": {
                          "type": "string",
                          "minLength": 1
                        },
                        "text": {
                          "type": "string"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      },
      "published": {
        "type": "boolean"
      },
      "publicationDate": {
        "type": "string",
        "format": "date"
      }
  },
  
  },
  "retry": 3,
  "retry_rules": {
    "invalid_structure": 2,
    "missing_field": 2,
    "invalid_type": 1
  }
}

@fixture
def client():
    base_shema = base_format["schema"]
    base_retry_count = base_format["retry"]
    base_retry_rules = base_format["retry_rules"]
    base_validator_id = base_format["validator_id"]

    json_validator = JSONValidator(json_agent.id, base_validator_id, base_shema, base_retry_count, base_retry_rules)
    xml_validator = XMLValidator(xml_agent.id, base_validator_id, base_shema, base_retry_count, base_retry_rules)
    yaml_validator = YAMLValidator(yaml_agent.id, base_validator_id, base_shema, base_retry_count, base_retry_rules)

    extra_shema = extra_format["schema"]
    extra_retry_count = extra_format["retry"]
    extra_retry_rules = extra_format["retry_rules"]
    extra_validator_id = extra_format["validator_id"]
    extra_json_validator = JSONValidator(extra_json_agent.id, extra_validator_id, extra_shema, extra_retry_count, extra_retry_rules)

    dotenv.load_dotenv('./.env')
    openai_key = os.getenv("OPENAI_API_KEY")
    lamoom = Lamoom(openai_key=openai_key)
    return lamoom

def test_json(client: Lamoom):
    response = client.call_and_validate(
        json_agent.id,             
        {},              
        'openai/o3-mini',
    )
    assert not response[-1].errors

def test_xml(client: Lamoom):
    response = client.call_and_validate(
        xml_agent.id,             
        {},              
        'openai/o3-mini',
    )
    assert not response[-1].errors

def test_yaml(client: Lamoom):
    response = client.call_and_validate(
        yaml_agent.id,             
        {},              
        'openai/o3-mini',
    )
    assert not response[-1].errors

def test_extra_json(client: Lamoom):
    response = client.call_and_validate(
        extra_json_agent.id,             
        {},              
        'openai/o3-mini',
    )
    assert not response[-1].errors

def test_two_json(client: Lamoom):
    response = client.call_and_validate(
        two_json_agent.id,             
        {},              
        'openai/o3-mini',
    )
    assert not response[-1].errors