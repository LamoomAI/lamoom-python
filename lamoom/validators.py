from abc import ABC, abstractmethod
import typing as t
import json
import jsonschema
import yaml
import xmltodict
from lamoom.response_parsers.response_parser import _get_format_from_response, Tag
from lamoom import AIResponse

class Validator(ABC):
    def __init__(self, validator_type: str, validation_format: t.Dict):
        self.type = validator_type
        self.schema = validation_format.get('schema')
        self.retry = int(validation_format.get('retry', 0))
        self.retry_rules = validation_format.get('retry_rules', {})
        self.errors = []

    @abstractmethod
    def validate(self, response: str) -> None:
        pass

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

class JSONValidator(Validator):
    def __init__(self, validation_format: t.Dict):
        super().__init__("json", validation_format)

    def validate(self, response: AIResponse):
        content, start_ind, end_ind = _get_format_from_response(
            response,
            [Tag("```json", "\n```", 0), Tag("```json", "```", 0)],
            start_from=0
        )
        if content:
            try:
                json_data = json.loads(content)
                jsonschema.validate(instance=json_data, schema=self.schema)
            except json.JSONDecodeError as e:
                self.add_error({
                    "type": "invalid_structure",
                    "details": f"Invalid JSON format at line {e.lineno}: {e.msg}"
                })
            except jsonschema.ValidationError as e:
                path = list(e.path) 
                error_location = " -> ".join(map(str, path))
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
        else:
            self.add_error({
                "type": "invalid_structure",
                "details": "No JSON content found in the response. "
                        "Expected JSON content between specified tags (e.g., '```json' and '```')."
            })


class YAMLValidator(Validator):
    def __init__(self, validation_format: t.Dict):
        super().__init__("yaml", validation_format)

    def validate(self, response: AIResponse):
        content, start_ind, end_ind = _get_format_from_response(
            response,
            [Tag("```yaml", "\n```", 0), Tag("```yaml", "```", 0)],
            start_from=0
        )

        if content:
            try:
                yaml_data = yaml.safe_load(content)  
                jsonschema.validate(instance=yaml_data, schema=self.schema) 
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
            except jsonschema.ValidationError as e:
                path = list(e.path) 
                error_location = " -> ".join(map(str, path))
                if e.validator == "required":

                    available_fields = list(yaml_data.keys())

                    current_instance = yaml_data
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
        else:
            self.add_error({
                "type": "invalid_structure",
                "details": "No YAML content found in the response. "
                           "Expected YAML content between specified tags (e.g., '```yaml' and '```')."
            })


class XMLValidator(Validator):
    def __init__(self, validation_format: t.Dict):
        super().__init__("xml", validation_format)

    def validate(self, response: AIResponse):
        content, start_ind, end_ind = _get_format_from_response(
            response,
            [Tag("```xml", "\n```", 0), Tag("```xml", "```", 0)],
            start_from=0
        )

        if content:
            try:
                content.replace('&', '&amp')
                xml_dict = xmltodict.parse(content)
                updated_dict = xml_dict.get(list(xml_dict.keys())[0], {})
                updated_dict = self.handle_lists(updated_dict)
                jsonschema.validate(instance=updated_dict, schema=self.schema)

            except xmltodict.expat.ExpatError as e:
                self.add_error({
                    "type": "invalid_structure",
                    "details": f"XML parsing error: {e.args[0]}"
                })
            except jsonschema.ValidationError as e:
                path = list(e.path) 
                error_location = " -> ".join(map(str, path))
                if e.validator == "required":

                    available_fields = list(updated_dict.keys())

                    current_instance = updated_dict
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
        else:
            self.add_error({
                "type": "invalid_structure",
                "details": "No XML content found."
            })
            return

    def handle_lists(self, data: t.Any) -> t.Dict:
        if isinstance(data, dict):
            for key, value in data.items():  
                if isinstance(value, dict) and (len(value.keys())==1):
                    children_list = value[list(value.keys())[0]]
                    data[key] = [self.handle_lists(item) for item in children_list]
                else:
                    data[key] = self.handle_lists(value)
        return data
