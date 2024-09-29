import json
import typing

import google.generativeai as genai

from model_server.model_endpoint.model_endpoints import ModelEndpoint
from model_server.model_endpoint.retryable_model import RetryableModel
from model_server.train_conn.server_config_props import GeminiModelEndpoint
from model_server.train_conn.server_config_props import ModelServerConfigProps

from python_di.configs.prototype import prototype_scope_bean, prototype_factory


@prototype_scope_bean(bindings=[ModelEndpoint])
class GeminiEndpoint(ModelEndpoint, RetryableModel):

    @prototype_factory()
    def __init__(self, model_server_props: ModelServerConfigProps):
        self.model_server_props: ModelServerConfigProps = model_server_props
        self.gemini_model: typing.Optional[genai.GenerativeModel] = None
        self._gemini: typing.Optional[GeminiModelEndpoint] = None

    @property
    def endpoint(self) -> str:
        return self._gemini.model_endpoint

    @property
    def gemini(self) -> typing.Optional[GeminiModelEndpoint]:
        return self._gemini

    @gemini.setter
    def gemini(self, gemini: typing.Optional[GeminiModelEndpoint]):
        self._gemini = gemini
        genai.configure(api_key=gemini.api_key)
        self.gemini_model = genai.GenerativeModel(gemini.gemini_model)

    def do_model(self, input_data: dict[str, str]):
        return self.gemini_model.generate_content(input_data['prompt']).text

    def parse_model_response(self, in_value):
        return self.parse_as_json(in_value)

    @staticmethod
    def parse_as_json(content):
        json_content = content
        middle_json = json_content.split('```json')
        if len(middle_json) == 1:
            json.loads(middle_json)
            return middle_json
        else:
            replaced_value = middle_json[1].split('```')[0]
            json.loads(replaced_value)
            return replaced_value

