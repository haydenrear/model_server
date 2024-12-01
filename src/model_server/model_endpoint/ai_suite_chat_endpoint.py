import json
import sys
import typing

from aisuite.framework import ChatCompletionResponse
from aisuite.framework.chat_provider import ChatProvider, ChatProviderInterface
from aisuite.provider import ProviderFactory
from model_server.model_endpoint.retryable_model import RetryableModel
from python_di.configs.prototype import prototype_scope_bean, prototype_factory

try:
    sys.path.append("/Users/hayde/IdeaProjects/drools/model_server/src")
except:
    pass

from model_server.train_conn.server_config_props import AiSuiteModelEndpoint
from model_server.model_endpoint.model_endpoints import ModelEndpoint
from model_server.train_conn.server_config_props import ModelServerConfigProps


@prototype_scope_bean(bindings=[ModelEndpoint[...]])
class AiSuiteChatEndpoint(ModelEndpoint, RetryableModel):

    @prototype_factory()
    def __init__(self, model_server_props: ModelServerConfigProps):
        self.model_server_props: ModelServerConfigProps = model_server_props
        self.provider: typing.Optional[ChatProvider] = None
        self._ai_suite: typing.Optional[AiSuiteModelEndpoint] = None

    @property
    def endpoint(self) -> str:
        return self._ai_suite.model_endpoint

    @property
    def ai_suite(self) -> typing.Optional[AiSuiteModelEndpoint]:
        return self._ai_suite

    @ai_suite.setter
    def ai_suite(self, ai_suite: typing.Optional[AiSuiteModelEndpoint]):
        self._ai_suite = ai_suite
        self.provider = ProviderFactory.create_chat_provider(ai_suite.provider_key, ai_suite.__dict__)

    def do_model(self, input_data: dict[str, ...]):
        if isinstance(self.provider, ChatProvider):
            return self.parse_open_ai_chat_response(self.provider.chat_completions_create(self.ai_suite.model, **input_data))
        elif isinstance(self.provider, ChatProviderInterface):
            return self.parse_open_ai_chat_response(self.provider.chat_completion_create(self.ai_suite.model, **input_data))

        raise NotImplementedError

    def parse_model_response(self, in_value):
        return self.parse_as_json(in_value)

    @staticmethod
    def parse_open_ai_chat_response(content: ChatCompletionResponse):
        c = [c.message.content for c in content.choices]
        return c[0] if len(c) == 1 else c

