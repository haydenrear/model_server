from rsocket.payload import Payload

from model_server.model_endpoint.model_endpoints import ModelEndpoint, InT, OutT
from python_di.configs.component import component


# @component(bind_to=[ModelEndpoint])
# class TestModelEndpoint(ModelEndpoint):
#     @property
#     def endpoint(self) -> str:
#         return 'test'
#
#     def do_model(self, input_data: Payload):
#         return 'hello world!'
#
# @component(bind_to=[ModelEndpoint])
# class TestPt(ModelEndpoint):
#
#     def do_model(self, input_data: InT) -> OutT:
#         pass
#
#     @property
#     def endpoint(self) -> str:
#         pass