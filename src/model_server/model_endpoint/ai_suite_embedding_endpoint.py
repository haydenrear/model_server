import sys
import typing

from aisuite.framework.chat_provider import ChatProvider, ChatProviderInterface
from aisuite.framework.embedding_provider import EmbeddingProvider, EmbeddingProviderInterface
from aisuite.provider import ProviderFactory
from python_di.configs.prototype import prototype_scope_bean, prototype_factory

try:
    sys.path.append("/Users/hayde/IdeaProjects/drools/model_server/src")
except:
    pass

from model_server.train_conn.server_config_props import AiSuiteModelEndpoint
from model_server.model_endpoint.model_endpoints import ModelEndpoint
from model_server.train_conn.server_config_props import ModelServerConfigProps


@prototype_scope_bean(bindings=[ModelEndpoint[...]])
class AiSuiteEmbeddingEndpoint(ModelEndpoint):

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
        self.provider = ProviderFactory.create_embedding_provider(ai_suite.provider_key, ai_suite.__dict__)

    def do_model(self, input_data: dict[str, ...]):
        messages = input_data['to_embed']
        model = self.ai_suite.model
        if 'model' in input_data.keys():
            model = input_data['model']
        if isinstance(self.provider, EmbeddingProvider):
            return self.provider.embedding_create(messages, model)
        elif isinstance(self.provider, EmbeddingProviderInterface):
            del input_data['model']
            created = self.provider.embedding_create(model=f"models/{model}", **input_data)
            input_data['model'] = model
            return created

        raise NotImplementedError

    def __call__(self, input_data: dict[str, ...]):
        return self.do_model(input_data)