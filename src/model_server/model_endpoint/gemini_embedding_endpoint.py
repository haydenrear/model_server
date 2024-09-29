import inspect
import json
import sys
import traceback
import typing

import injector
from attr.validators import instance_of
from fontTools.misc.plistlib import end_real
from pasta.base.codegen_test import AutoFormatTest


from docker.pasta.pasta.base.ast_utils import replace_child
from python_di.configs.component import component
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
import transformers
import torch

from python_di.configs.prototype import prototype_scope_bean, prototype_factory

import os
import google.generativeai as genai

try:
    sys.path.append("/Users/hayde/IdeaProjects/drools/model_server/src")
except:
    pass

from python_util.logger.logger import LoggerFacade
from model_server.model_endpoint.retryable_model import RetryableModel

from model_server.train_conn.server_config_props import GeminiModelEndpoint
from model_server.model_endpoint.model_endpoints import ModelEndpoint
from model_server.train_conn.server_config_props import ModelServerConfigProps, HuggingfaceModelEndpoint


@prototype_scope_bean(bindings=[ModelEndpoint])
class GeminiEmbeddingEndpoint(ModelEndpoint, RetryableModel):

    @prototype_factory()
    def __init__(self, model_server_props: ModelServerConfigProps):
        self.model_server_props: ModelServerConfigProps = model_server_props
        self.gemini_model = None
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
        self.gemini_model = lambda inputs: genai.embed_content(self.gemini.gemini_model, inputs['prompt'],
                                                               **{k:v for k,v in inputs.items() if k != 'prompt'})

    def do_model(self, input_data: dict[str, str]):
        if 'task_type' not in input_data.keys():
            input_data['task_type'] = 'retrieval_document'
        if 'title' not in input_data.keys():
            input_data['title'] = 'Generate Embedding for Computer Code'
        return self.gemini_model(input_data)


    def parse_model_response(self, in_value):
        return in_value["embedding"]
