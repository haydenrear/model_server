import abc
import inspect
import json
import typing

from python_util.logger.logger import LoggerFacade

class RetryableModel(abc.ABC):

    @abc.abstractmethod
    def do_model(self, in_prompt: str):
        pass

    @abc.abstractmethod
    def parse_model_response(self, in_value):
        pass

    def __call__(self, prompt_value, num_tries=0):
        if num_tries <= 5:
            if num_tries != 0:
                LoggerFacade.info(f"Trying to generate content again - on {num_tries} try.")
            content = self.do_model(prompt_value)
            try:
                LoggerFacade.debug("Loaded content from model.")
                replaced_value = self.parse_model_response(content)
                LoggerFacade.debug("Parsed content to response.")
                return {'data': replaced_value}
            except Exception as e:
                LoggerFacade.warn(f"Failed to parse response to JSON: {e}")
                return self(prompt_value, num_tries + 1)
        else:
            exc = ', '.join([i for i in self.walk_back_exc()])
            LoggerFacade.error(f"Could not generate content even after maximum number of tries with last exceptions: {exc}.")
            return {'data': "{" + f"""
                    "status": 503,
                    "message": "Failed to generate content successfully after 5 tries.",
                    "exception": "{exc}"
                """ + "}"}

    @classmethod
    def walk_back_exc(cls):
        b = inspect.currentframe().f_back
        exc = []
        for i in range(8):
            for j in filter(lambda x: isinstance(x, Exception) and str(x) not in exc, [i for i in b.f_locals.values()]):
                exc.append(str(j))
            b = b.f_back

        return exc