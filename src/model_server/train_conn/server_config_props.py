import os
import typing

from pydantic.main import BaseModel

from python_di.env.base_module_config_props import ConfigurationProperties
from python_di.properties.configuration_properties_decorator import configuration_properties


class HuggingfaceModelEndpoint(BaseModel):
    hf_model: str
    model_endpoint: str
    pipeline: dict[str, object]
    pipeline_kwargs: dict[str, object]


@configuration_properties(
    prefix_name='model_server'
)
class ModelServerConfigProps(ConfigurationProperties):
    port: int
    host: str
    hf_model_endpoint: dict[str, HuggingfaceModelEndpoint]
