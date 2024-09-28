import json
import unittest

from attr import dataclass
from flask import Flask

import sys


try:
    sys.path.append("/Users/hayde/IdeaProjects/drools/model_server/src")
except:
    pass

from model_server.train_conn.server_config_props import GeminiModelEndpoint, ModelServerConfigProps
from model_server.model_endpoint.gemini_endpoint import GeminiEndpoint
from model_server.train_conn.model_server_config import ServerRunnerConfig
from model_server.train_conn.server_runner import ServerRunnerProvider, ServerRunner, HttpServerRunner

from python_util.collections.collection_util import first_from_iter, first

from python_di.configs.bean import test_inject
from python_di.configs.test import test_booter, boot_test
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn

from unittest import mock

@test_booter(scan_root_module=ServerRunnerConfig)
class ServerRunnerBoot:
    pass

@boot_test(ctx=ServerRunnerBoot)
class GeminiModelEndpointTest(unittest.TestCase):

    gemini: GeminiEndpoint
    gemini_model_endpoint: ModelServerConfigProps

    @test_inject()
    @autowire_fn()
    def construct(self, gemini: GeminiEndpoint,
                  gemini_model_endpoint: ModelServerConfigProps):
        self.gemini = gemini
        self.gemini_model_endpoint = gemini_model_endpoint

    def test_gemini_model_endpoint(self):
        self.gemini.gemini = first(self.gemini_model_endpoint.gemini_model_endpoint)

        @dataclass(init=True)
        class Ret:
            text: str

        self.gemini.gemini_model.generate_content = mock.MagicMock(return_value = Ret("hello"))
        did_model = self.gemini({'prompt': 'hello'})

        assert did_model
        assert self.gemini
        loaded = json.loads(did_model['data'])

        assert loaded['status'] == 503
        assert loaded['message'] == "Failed to generate content successfully after 5 tries."
        assert "the JSON object must be str" in loaded['exception']

        self.gemini.gemini_model.generate_content = mock.MagicMock(return_value = Ret("```json\n{ \"someKey\": \"someValue\" }```"))
        did_model = self.gemini({'prompt': 'hello'})

        assert did_model
        assert self.gemini
        loaded = json.loads(did_model['data'])
        assert loaded['someKey'] == 'someValue'
