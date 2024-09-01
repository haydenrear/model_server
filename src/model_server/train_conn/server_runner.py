import asyncio
import http
import logging
import typing
from http.server import SimpleHTTPRequestHandler
from typing import Awaitable

import injector
from flask import app
from rsocket.frame_helpers import ensure_bytes
from rsocket.helpers import create_response
from rsocket.payload import Payload
from rsocket.routing.request_router import RequestRouter
from rsocket.routing.routing_request_handler import RoutingRequestHandler
from rsocket.rsocket_server import RSocketServer
from rsocket.transports.tcp import TransportTCP

from model_server.model_endpoint.model_endpoints import ModelEndpoint
from model_server.train_conn.huggingface_endpoints import HfEndpoint
from model_server.train_conn.server_config_props import ModelServerConfigProps, HuggingfaceModelEndpoint
from python_di.configs.component import component
from python_di.inject.profile_composite_injector.composite_injector import profile_scope
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn, InjectionDescriptor, \
    InjectionType
from python_util.logger.logger import LoggerFacade

from flask import Flask, request

app = Flask(__name__)


@component()
class ServerRunnerProvider:

    @injector.inject
    def __init__(self,
                 model_server_props: ModelServerConfigProps,
                 models: typing.List[ModelEndpoint]):
        self.base_embedding_models = models
        self.model_server_props = model_server_props

    def create_routes(self) -> RoutingRequestHandler:
        router = RequestRouter()

        for model_endpoints in self.base_embedding_models:
            LoggerFacade.info(f"Starting model endpoint {model_endpoints.endpoint}.")

            @router.response(model_endpoints.endpoint)
            async def model_embedding_endpoint(payload: Payload) -> Awaitable[Payload]:
                return create_response(ensure_bytes(model_endpoints.do_model(payload)))

        return RoutingRequestHandler(router)

    def start_server(self):
        def session(*connection):
            RSocketServer(TransportTCP(*connection), handler_factory=self.create_routes)

        return session


@component()
class ServerRunner:
    @injector.inject
    def __init__(self, model_server_props: ModelServerConfigProps,
                 server_runner_provider: ServerRunnerProvider):
        self.server_runner_provider = server_runner_provider
        self.model_server_props = model_server_props

    def run(self):
        asyncio.run(run_server(self.model_server_props.port,
                               self.model_server_props.host,
                               self.server_runner_provider.start_server()))


async def run_server(server_port: int, host: str, cxn):
    logging.info('Starting server at %s:%s', host, server_port)

    server = await asyncio.start_server(cxn, host, server_port)

    async with server:
        await server.serve_forever()

    # Request handler


class HttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Handle GET requests
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Hello, world!')


@component()
class HttpServerRunnerProvider:

    @injector.inject
    def __init__(self,
                 model_server_props: ModelServerConfigProps,
                 models: typing.List[ModelEndpoint]):
        self.base_embedding_models = models
        self.model_server_props = model_server_props
        self.did_create = False

    def create_routes(self):

        for model_endpoints in self.base_embedding_models:
            if isinstance(model_endpoints, HfEndpoint):
                continue
            LoggerFacade.info(f"Starting model endpoint {model_endpoints.endpoint}.")

            @app.route(model_endpoints.endpoint, methods=['GET', 'POST'])
            def serve():
                return model_endpoints.do_model(request.json)

        for k, hf in self.model_server_props.hf_model_endpoint.items():
            self.create_hf(e=hf)

        self.did_create = True

    @autowire_fn(
        descr={
            "e": InjectionDescriptor(InjectionType.Provided),
            "hf_model_endpoint": InjectionDescriptor(InjectionType.Dependency)
        }
    )
    def create_hf(self,
                  e: HuggingfaceModelEndpoint,
                  hf_model_endpoint: HfEndpoint):
        hf_model_endpoint.hf = e

        @app.route(e.model_endpoint, methods=['GET', 'POST'])
        def serve():
            return hf_model_endpoint.do_model(request.json)


@component()
class HttpServerRunner:
    @injector.inject
    def __init__(self, model_server_props: ModelServerConfigProps,
                 server_runner_provider: HttpServerRunnerProvider):
        self.server_runner_provider = server_runner_provider
        self.model_server_props = model_server_props

    def run(self):
        self.server_runner_provider.create_routes()
        app.run(port=self.model_server_props.port)


async def run_http_server(server_port: int, host: str, cxn):
    logging.info('Starting server at %s:%s', host, server_port)

    server = await asyncio.start_server(cxn, host, server_port)

    async with server:
        await server.serve_forever()
