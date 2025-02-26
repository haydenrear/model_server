import json
import sys
import typing

import rerankers
from rerankers.results import RankedResults

from aisuite.framework import ChatCompletionResponse
from aisuite.framework.chat_provider import ChatProvider, ChatProviderInterface
from aisuite.framework.rerank_provider import RerankProvider, RerankProviderInterface
from aisuite.provider import ProviderFactory, ProviderType
from model_server.model_endpoint.ai_suite_chat_endpoint import AiSuiteChatEndpoint
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
class AiSuiteRerankEndpoint(ModelEndpoint, RetryableModel):

    @prototype_factory()
    def __init__(self, model_server_props: ModelServerConfigProps):
        self.model_server_props: ModelServerConfigProps = model_server_props
        self.provider: typing.Optional[RerankProvider] = None
        self._ai_suite: typing.Optional[AiSuiteModelEndpoint] = None

    @property
    def endpoint(self) -> str:
        return self._ai_suite.model_endpoint

    @property
    def ai_suite(self) -> typing.Optional[AiSuiteModelEndpoint]:
        return self._ai_suite

    @ai_suite.setter
    def ai_suite(self, ai_suite: typing.Optional[AiSuiteModelEndpoint]):
        ai_suite.provider_type = ProviderType.RERANK
        self._ai_suite = ai_suite
        self.provider = ProviderFactory.create_rerank_provider(ai_suite.provider_key, ai_suite.__dict__)


    def __call__(self, data: dict[str, ...], **kwargs) -> rerankers.results.RankedResults:
        return self.parse_model_response(self.do_model(data, **kwargs))

    def do_model(self, data: dict[str, ...], **kwargs):
        data = data['rerank_body']
        if isinstance(self.provider, RerankProvider):
            return self.provider.rerank_create(self.ai_suite.model, **kwargs)(data)
        elif isinstance(self.provider, RerankProviderInterface):
            return self.provider.rerank_create(self.ai_suite.model, **kwargs)(data)

        raise NotImplementedError

    def parse_model_response(self, in_value: RankedResults):
        results = {
            r.rank: self._parse_json_doc(r.document, r.rank, r.score)
            for i, r in enumerate(in_value.results)
        }

        return {
            'ranked_results': results,
            'query': in_value.query
        }

    def _parse_json_doc(self, r: rerankers.Document, rank: typing.Optional[int] = None,
                        score: typing.Optional = None):
        parsed_doc = {}
        self._add_if_exists(r, 'text', parsed_doc)
        self._add_if_exists(r, 'metadata', parsed_doc)
        self._add_if_exists(r, 'doc_id', parsed_doc)
        self._add_if_exists(r, 'score', parsed_doc)
        self._add_if_exists(r, 'document_type', parsed_doc)
        self._add_if_exists(r, 'metadata', parsed_doc)
        self._add_value_if_exists(rank, 'rank', parsed_doc)

        if 'score' not in parsed_doc.keys() or parsed_doc['score'] <= 0.0:
            self._add_value_if_exists(score, 'score', parsed_doc)

        return parsed_doc

    def _add_if_exists(self, r: rerankers.Document, key_to_add: str, to_add_to: dict[str, ...]):
        if hasattr(r, key_to_add):
            attr = getattr(r, key_to_add)
            if attr:
                to_add_to[key_to_add] = attr

        return to_add_to

    def _add_value_if_exists(self, value_to_add, key_to_add: str, to_add_to: dict[str, ...]):
        if value_to_add is not None and key_to_add is not None:
            to_add_to[key_to_add] = value_to_add

        return to_add_to
