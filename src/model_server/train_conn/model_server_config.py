from model_server.model_endpoint.huggingface_endpoints import HfEndpoint
from model_server.train_conn.server_config_props import ModelServerConfigProps
from model_server.train_conn.server_runner import ServerRunnerProvider
from python_di.configs.component_scan import component_scan
from python_di.configs.di_configuration import configuration
from python_di.configs.enable_configuration_properties import enable_configuration_properties


@configuration()
@enable_configuration_properties(config_props=[ModelServerConfigProps])
@component_scan(base_classes=[HfEndpoint, ServerRunnerProvider])
class ServerRunnerConfig:
    pass


