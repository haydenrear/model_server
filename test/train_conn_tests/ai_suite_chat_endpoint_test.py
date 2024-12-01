import json
import logging
import unittest

from attr import dataclass
from drools_py.configs.config import ConfigType
from flask import Flask

import sys

from model_server.model_endpoint.ai_suite_chat_endpoint import AiSuiteChatEndpoint
from python_util.logger.log_level import LogLevel
from src.model_server.train_conn.server_config_props import ModelType

try:
    sys.path.append("/Users/hayde/IdeaProjects/drools/model_server/src")
except:
    pass

from model_server.train_conn.server_config_props import GeminiModelEndpoint, ModelServerConfigProps
from model_server.model_endpoint.gemini_endpoint import GeminiEndpoint
from model_server.train_conn.model_server_config import ServerRunnerConfig
from model_server.train_conn.server_runner import ServerRunnerProvider, ServerRunner, HttpServerRunner

from python_util.collections.collection_util import first_from_iter, first, first_matching

from python_di.configs.bean import test_inject
from python_di.configs.test import test_booter, boot_test
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn

from unittest import mock

LogLevel.set_log_level(logging.DEBUG)

@test_booter(scan_root_module=ServerRunnerConfig)
class ServerRunnerBoot:
    pass

@boot_test(ctx=ServerRunnerBoot)
class AiSuiteChatEndpointTest(unittest.TestCase):

    ai_suite: AiSuiteChatEndpoint
    model_server_props: ModelServerConfigProps

    @test_inject()
    @autowire_fn()
    def construct(self,
                  ai_suite: AiSuiteChatEndpoint,
                  gemini_model_endpoint: ModelServerConfigProps):
        self.ai_suite = ai_suite
        self.model_server_props = gemini_model_endpoint

    def test_ai_suite_model_endpoint(self):
        self.ai_suite.ai_suite = self.model_server_props.ai_suite_model_endpoint.get('')

        @dataclass(init=True)
        class Ret:
            text: str
        # TODO:
        # self.gemini.gemini_model.generate_content = mock.MagicMock(return_value = Ret("hello"))
        # did_model = self.gemini({'prompt': 'hello'})
        #
        # assert did_model
        # assert self.gemini
        # loaded = json.loads(did_model['data'])
        #
        # assert loaded['status'] == 503
        # assert loaded['message'] == "Failed to generate content successfully after 5 tries."
        # assert "the JSON object must be str" in loaded['exception']
        #
        # self.gemini.gemini_model.generate_content = mock.MagicMock(return_value = Ret("```json\n{ \"someKey\": \"someValue\" }```"))
        # did_model = self.gemini({'prompt': 'hello'})
        #
        # assert did_model
        # assert self.gemini
        # loaded = json.loads(did_model['data'])
        # assert loaded['someKey'] == 'someValue'
