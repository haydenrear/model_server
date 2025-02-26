import dataclasses
import json
import logging
import os
import unittest

import rerankers
from attr import dataclass
from rerankers.results import RankedResults

from aisuite.framework import ChatCompletionResponse
from aisuite.framework.choice import Choice
from aisuite.framework.message import Message
from aisuite.providers.googlecloud_provider import GooglecloudReranker, GOOGLE_APP_CRED_KEY
from drools_py.configs.config import ConfigType
from flask import Flask

import sys

from model_server.model_endpoint.ai_suite_chat_endpoint import AiSuiteChatEndpoint
from model_server.model_endpoint.ai_suite_rerank_endpoint import AiSuiteRerankEndpoint
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
class AiSuiteRerankEndpointTest(unittest.TestCase):

    ai_suite: AiSuiteRerankEndpoint
    model_server_props: ModelServerConfigProps

    @test_inject()
    @autowire_fn()
    def construct(self,
                  ai_suite: AiSuiteRerankEndpoint,
                  gemini_model_endpoint: ModelServerConfigProps):
        AiSuiteRerankEndpointTest.ai_suite = ai_suite
        AiSuiteRerankEndpointTest.model_server_props = gemini_model_endpoint

    def test_ai_suite_rerank_endpoint(self):
        self.ai_suite.ai_suite = self.model_server_props.ai_suite_model_endpoint.get('google_genai_rerank')

        @dataclasses.dataclass(init=True)
        class ContentItem:
            content: str
        @dataclasses.dataclass(init=True)
        class Ret:
            records: list[ContentItem]

        if GOOGLE_APP_CRED_KEY not in os.environ.keys():
            os.environ[GOOGLE_APP_CRED_KEY] = "fake"

        mock_client = Ret([])
        mock_client.ranking_config_path = mock.MagicMock(return_value = "")
        reranker = GooglecloudReranker("model", "project", "app_cred", mock_client)
        reranker.gen_ai = mock.MagicMock(return_value=Ret([ContentItem("hello") for i in range(10)]))
        did_model = self.ai_suite(query="okay then...", docs=["hello" for i in range(20)], client=mock_client)

        assert did_model
        assert self.ai_suite
        loaded = did_model['ranked_results']

        assert len(loaded) == 20

        for i, (k, v) in enumerate(loaded.items()):
            assert v['document']['doc_id'] == str(i)
            assert v['document']['rank'] == i
            assert v['document']['document_type'] == 'text'
            assert v['document']['text'] == 'hello'

