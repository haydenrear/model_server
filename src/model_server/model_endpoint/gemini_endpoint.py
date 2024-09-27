import inspect
import json
import traceback
import typing

import injector
from attr.validators import instance_of
from fontTools.misc.plistlib import end_real
from pasta.base.codegen_test import AutoFormatTest

from model_server.model_endpoint.model_endpoints import ModelEndpoint
from model_server.train_conn.server_config_props import ModelServerConfigProps, HuggingfaceModelEndpoint

from docker.pasta.pasta.base.ast_utils import replace_child
from python_di.configs.component import component
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
import transformers
import torch

from python_di.configs.prototype import prototype_scope_bean, prototype_factory

import os
import google.generativeai as genai

from python_util.logger.logger import LoggerFacade
from src.model_server.train_conn.server_config_props import GeminiModelEndpoint


@prototype_scope_bean(bindings=[ModelEndpoint])
class GeminiEndpoint(ModelEndpoint):

    @prototype_factory()
    def __init__(self, model_server_props: ModelServerConfigProps):
        self.model_server_props: ModelServerConfigProps = model_server_props
        self._gemini: typing.Optional[GeminiModelEndpoint] = None
        self.gemini_model: typing.Optional[genai.GenerativeModel] = None

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
        return self.do_model_recursive(input_data['prompt'])

    def do_model_recursive(self, prompt_value, num_tries=0):
        if num_tries <= 5:
            if num_tries != 0:
                LoggerFacade.info(f"Trying to generate content again - on {num_tries} try.")
            content = self.gemini_model.generate_content(prompt_value)
            try:
                LoggerFacade.debug("Loaded content to JSON object.")
                replaced_value = self.parse_as_json(content)
                return {'data': replaced_value}
            except Exception as e:
                LoggerFacade.warn(f"Failed to parse response to JSON: {e}")
                return self.do_model_recursive(prompt_value, num_tries + 1)
        else:
            exc = ''.join([str(i) for i
                           in filter(lambda x: isinstance(x, Exception),
                                     inspect.currentframe().f_back.f_locals.values())])
            LoggerFacade.error(f"Could not generate content even after maximum number of tries with last exception: {exc}.")
            return {'data': "{" + f"""
                "status": 503,
                "message": "Failed to generate JSON content successfully after 5 tries.",
                "exception": "{exc}"
            """ + "}"}

    @staticmethod
    def parse_as_json(content):
        json_content = content.text
        middle_json = json_content.split('```json')
        if len(middle_json) == 1:
            json.loads(middle_json)
            return middle_json
        else:
            replaced_value = middle_json[1].split('```')[0]
            json.loads(replaced_value)
            return replaced_value

