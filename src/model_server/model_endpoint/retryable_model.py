import abc
import json

from python_util.logger.inspect_utils import walk_back_exc
from python_util.logger.logger import LoggerFacade


class RetryableModel(abc.ABC):

    @abc.abstractmethod
    def do_model(self, in_prompt: str):
        pass

    @abc.abstractmethod
    def parse_model_response(self, in_value):
        pass


    @staticmethod
    def parse_as_json(content):
        json_content = content
        middle_json = json_content.split('```json')
        if len(middle_json) == 1:
            json_value = middle_json[0]
            json_value = RetryableModel.remove_ending_delim(json_value)
            loaded_json = json.loads(json_value)
            return loaded_json
        else:
            json_value = middle_json[1]
            json_value = RetryableModel.remove_ending_delim(json_value)
            if '```java' in json_value:
                all_json = RetryableModel.parse_by_language(json_value, 'java')
            elif '```python' in json_value:
                all_json = RetryableModel.parse_by_language(json_value, 'python')
            else:
                all_json = json_value
            loaded_json = json.loads(all_json)
            return loaded_json

    @staticmethod
    def remove_ending_delim(json_value):
        json_value = json_value.strip()
        if json_value.endswith('```'):
            json_value = json_value[:-3]
        return json_value

    @staticmethod
    def parse_by_language(json_value: str, language: str):
        splitted_again = json_value.split("```" + language)
        if len(splitted_again) == 1:
            return splitted_again[0]

        first_part_of_json = splitted_again[0]
        json_value = splitted_again[1]
        json_value = json_value.strip()
        second_json_split_again = json_value.split('```')
        json_value: str = second_json_split_again[0]
        json_value = json_value.replace('\\n', '\\\\n').replace('\n', '\\\n').replace('"', "\"")

        if len(second_json_split_again) == 1:
            return first_part_of_json + json_value
        else:
            last_part_of_json = second_json_split_again[1]
            return first_part_of_json + json_value + last_part_of_json

    def __call__(self, prompt_value, num_tries=0, max_tries=5, last_exc = None):
        if num_tries < max_tries:
            if num_tries != 0:
                LoggerFacade.info(f"Trying to generate content again - on {num_tries} try.")
            try:
                if last_exc is not None:
                    to_send = f'Failed to parse JSON response from model for the prompt at end of this request with exception\n\n{last_exc}\n\nPlease try to respond with valid JSON\n\n{prompt_value}'
                    content = self.do_model(to_send)
                else:
                    content = self.do_model(prompt_value)

                LoggerFacade.debug("Loaded content from model.")
                replaced_value = self.parse_model_response(content)
                LoggerFacade.debug("Parsed content to response.")

                return {'data': replaced_value}
            except Exception as e:
                LoggerFacade.warn(f"Failed to parse response to JSON: {e}")
                return self(prompt_value, num_tries + 1)
        else:
            exc = ', '.join([i for i in walk_back_exc(num_frames_walk_back=8)])
            LoggerFacade.error(f"Could not generate content even after maximum number of tries with last exceptions: {exc}.")
            return {
                'data': {
                    'status': 503,
                    'message': f"Failed to generate content successfully after {num_tries} tries.",
                    "exception": f"{exc}"
                }
            }
