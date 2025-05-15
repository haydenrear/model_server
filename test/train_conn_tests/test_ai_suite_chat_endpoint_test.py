import dataclasses
import logging
import sys
import unittest

from aisuite.framework import ChatCompletionResponse
from aisuite.framework.choice import Choice
from aisuite.framework.message import Message

from model_server.model_endpoint.ai_suite_chat_endpoint import AiSuiteChatEndpoint
from python_util.logger.log_level import LogLevel

try:
    sys.path.append("/Users/hayde/IdeaProjects/drools/model_server/src")
except:
    pass

from model_server.train_conn.server_config_props import ModelServerConfigProps
from model_server.train_conn.model_server_config import ServerRunnerConfig

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
        AiSuiteChatEndpointTest.ai_suite = ai_suite
        AiSuiteChatEndpointTest.model_server_props = gemini_model_endpoint
        print(self.model_server_props)

    def test_ai_suite_model_endpoint(self):
        self.ai_suite.ai_suite = self.model_server_props.ai_suite_model_endpoint.get('gemini_flash')

        @dataclasses.dataclass(init=True)
        class Ret:
            value: str
        self.ai_suite.provider.chat_completions_create = mock.MagicMock(return_value=Ret("hello"))
        did_model = self.ai_suite({'prompt': 'hello'})

        assert did_model
        assert self.ai_suite
        loaded = did_model['data']

        assert loaded['status'] == 503
        assert loaded['message'] == "Failed to generate content successfully after 5 tries."
        assert "'Ret' object has no attribute 'choices'" in loaded['exception']

        response = ChatCompletionResponse()
        choice = Choice()
        message = Message()
        message.content = "```json\n{ \"someKey\": \"someValue\" }```"
        choice.message = message
        response.choices = [choice]

        self.ai_suite.provider.chat_completions_create = mock.MagicMock(return_value = response)
        did_model = self.ai_suite({'prompt': 'hello'})

        assert did_model
        assert self.ai_suite
        loaded = did_model['data']
        assert loaded['someKey'] == 'someValue'
