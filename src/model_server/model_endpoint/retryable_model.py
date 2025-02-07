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
            return json.loads(middle_json)
        else:
            replaced_value = middle_json[1].split('```')[0].strip('\'')
            return json.loads(replaced_value)


    def __call__(self, prompt_value, num_tries=0, max_tries=5):
        if num_tries < max_tries:
            if num_tries != 0:
                LoggerFacade.info(f"Trying to generate content again - on {num_tries} try.")
            try:
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
