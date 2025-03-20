from abc import ABC, abstractmethod
import typing as t
import json
import jsonschema
import yaml
import xmltodict
from lamoom.settings import PROMPT_VALIDATORS
from lamoom import AIResponse

class Validator(ABC):
    def __init__(self, validator_id: str, validator_type: str, schema: t.Dict = None, retry_count: int = 0, retry_rules: t.Optional[t.Dict]= None):
        self.id = validator_id
        self.type = validator_type
        if schema:
            self.schema = schema
        else:
            self.schema = {}
        self.retry = retry_count
        if retry_rules:
            self.retry_rules = retry_rules
        else:
            self.retry_rules = {}
        self.errors = []

    @abstractmethod
    def validate(self, response: str) -> None:
        pass

    def add_field_to_validate(self, field: str, field_type: t.Optional[t.Union[str, t.Dict]], **kwargs) -> None:

        field_type_str = isinstance(field_type, str)
        field_type_dict = isinstance(field_type, dict)
        if self.schema:
            if field not in self.schema["required"]:
                self.schema["required"].append(field)
            if field_type_str:
                self.schema["properties"][field] = {"type": field_type}
            elif field_type_dict:
                self.schema["properties"][field] = field_type
        else:
            self.schema = {
                "type": "object",
                "required": [field],
            }
            if field_type_str:
                self.schema["properties"] = {
                    field: {
                        "type": field_type
                    }
                }
            elif field_type_dict:
                self.schema["properties"] = {
                    field: field_type
                }
        if field_type_str:
            for key, value in kwargs.items():
                self.schema["properties"][field][key] = value


    def attach_to_promt(self, prompt_id: str) -> None:
        if prompt_id in PROMPT_VALIDATORS:
            PROMPT_VALIDATORS[prompt_id][self.id] = self
        else:
            PROMPT_VALIDATORS[prompt_id] = {self.id: self}

    def clear_errors(self) -> None:
        self.errors = []

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def get_errors(self) -> t.List[t.Dict]:
        return self.errors

    def add_error(self, error: t.Dict):
        self.errors.append(error)

    def update_format(self, validator_type:str, validation_format: t.Dict):
        self.__init__(validator_type, validation_format)

    def can_retry(self) -> bool:
        error_type = self.errors[0].get("type")
        if error_type:
            if int(self.retry_rules.get(error_type, 0)) > 0:
                self.retry_rules[error_type] -= 1
                return True
        return False

    def format_error(self, error: t.Dict) -> str:
        """Format the error into a human-readable string."""
        error_type = error.get("type", "unknown_error")
        if error_type == "invalid_structure":
            return f"Error parsing JSON: {error.get('details', 'Unknown parsing error')}. Please check your JSON format."
        elif error_type == "missing_field":
            return (
                f"Missing required field: {error.get('field', 'Unknown field')}. "
                f"Available fields are: {error.get('available_fields', 'N/A')}. "
                f"Error location: {error.get('error_location', 'Unknown location')}."
            )
        elif error_type == "invalid_type":
            return (
                f"Invalid type for field: {error.get('field', 'Unknown field')}. "
                f"Expected type: {error.get('expected_type', 'N/A')}, but got: {error.get('actual_type', 'N/A')}."
                f"Error location: {error.get('error_location', 'Unknown location')}."
            )
        elif error_type == "invalid_schema":
            return (
                f"Schema error encountered: {error.get('details', 'Unknown error')}.\n"
                f"Error location in schema: {error.get('error_location', 'Unknown location')}."
            )
        return "An unknown error occurred."


class FormatValidator(Validator):
    def validate(self, json_data: t.Dict):
        try:
            jsonschema.validate(instance=json_data, schema=self.schema)
        except jsonschema.ValidationError as e:
            path = list(e.path)
            if path:
                error_location = " -> ".join(map(str, path))
            else:
                error_location = "/"
            if e.validator == "required":

                available_fields = list(json_data.keys())

                current_instance = json_data
                for key in path:
                    if isinstance(key, int):
                        current_instance = current_instance[key-1]
                    elif isinstance(current_instance, dict) and key in current_instance:
                        current_instance = current_instance[key]
                    else:
                        current_instance = None
                        break

                if isinstance(current_instance, dict):
                    available_fields = list(current_instance.keys())

                self.add_error({
                    "type": "missing_field",
                    "field": e.args[0].split()[0],
                    "available_fields": available_fields,
                    "error_location": error_location
                })
            elif e.validator == "type":
                self.add_error({
                    "type": "invalid_type",
                    "field": e.args[0],
                    "expected_type": e.schema.get("type", "unknown"),
                    "actual_type": type(e.instance).__name__,
                    "error_location": error_location
                })
        except jsonschema.SchemaError as e:
            path = list(e.context[0].context[0].absolute_path) 
            error_location = " -> ".join(map(str, path))
            context = e.context[0].context[0].context[0].args[0]
            self.add_error({
                "type": "invalid_schema",
                "details": f"Schema error: {context}",
                "error_location": error_location
            })
    
class JSONValidator(FormatValidator):
    def __init__(self, validator_id: str, schema: t.Dict = None, retry_count: int = 0, retry_rules: t.Optional[t.Dict]= None):
        super().__init__(validator_id, "json", schema, retry_count, retry_rules)

    def validate(self, response: AIResponse):
        json_list = response.json_list
        if json_list:
            for _json in json_list:
                try:
                    json_data = json.loads(_json["content"])  
                except json.JSONDecodeError as e:
                    self.add_error({
                        "type": "invalid_structure",
                        "details": f"Invalid JSON format at line {e.lineno}: {e.msg}"
                    })
                    return
                super().validate(json_data)
            if len(self.get_errors()) < len(json_list):
                self.clear_errors()
        else:
            self.add_error({
                "type": "invalid_structure",
                "details": "No JSON content found in the response. "
                        "Expected JSON content between specified tags (e.g., '```json' and '```')."
            })


class YAMLValidator(FormatValidator):
    def __init__(self, validator_id: str, schema: t.Dict = None, retry_count: int = 0, retry_rules: t.Optional[t.Dict]= None):
        super().__init__(validator_id, "yaml", schema, retry_count, retry_rules)

    def validate(self, response: AIResponse):
        yaml_list = response.yaml_list
        if yaml_list:
            for _yaml in yaml_list:
                try:
                    yaml_data = yaml.safe_load(_yaml["content"])  
                except yaml.YAMLError as e:
                    if hasattr(e, 'problem_mark'):
                        line = e.problem_mark.line + 1
                        column = e.problem_mark.column + 1 
                        self.add_error({
                            "type": "invalid_structure",
                            "details": f"Invalid YAML format at line {line}, column {column}: {str(e)}"
                        })
                    else:
                        self.add_error({
                            "type": "invalid_structure",
                            "details": f"Invalid YAML format: {str(e)}"
                        })
                    return
                super().validate(yaml_data)
            if len(self.get_errors()) < len(yaml_list):
                self.clear_errors()
        else:
            self.add_error({
                "type": "invalid_structure",
                "details": "No YAML content found in the response. "
                           "Expected YAML content between specified tags (e.g., '```yaml' and '```')."
            })


class XMLValidator(FormatValidator):
    def __init__(self, validator_id: str, schema: t.Dict = None, retry_count: int = 0, retry_rules: t.Optional[t.Dict]= None):
        super().__init__(validator_id, "xml", schema, retry_count, retry_rules)

    def validate(self, response: AIResponse):
        xml_list = response.xml_list
        if xml_list:
            for _xml in xml_list:
                try:
                    content = _xml["content"]
                    content.replace('&', '&amp')
                    xml_dict = xmltodict.parse(content)
                    updated_dict = xml_dict.get(list(xml_dict.keys())[0], {})
                    updated_dict = self._handle_lists(updated_dict)
                except xmltodict.expat.ExpatError as e:
                    self.add_error({
                        "type": "invalid_structure",
                        "details": f"XML parsing error: {e.args[0]}"
                    })
                    return
                super().validate(updated_dict)
            if len(self.get_errors()) < len(xml_list):
                self.clear_errors()
        else:
            self.add_error({
                "type": "invalid_structure",
                "details": "No XML content found."
            })
            return

    def _handle_lists(self, data: t.Any) -> t.Dict:
        if isinstance(data, dict):
            for key, value in data.items():  
                if isinstance(value, dict) and (len(value.keys())==1):
                    children_list = value[list(value.keys())[0]]
                    data[key] = [self._handle_lists(item) for item in children_list]
                else:
                    data[key] = self._handle_lists(value)
        return data
