import abc
import enum
import os
import typing

from pydantic.main import BaseModel

from aisuite.provider import ProviderType
from python_di.env.base_module_config_props import ConfigurationProperties
from python_di.properties.configuration_properties_decorator import configuration_properties

class AiSuiteModel(BaseModel, abc.ABC):
    pass
#     @property
#     @abc.abstractmethod
#     def model(self):
#         return
#
#     @model.setter
#     @abc.abstractmethod
#     def model(self, model: str):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def model_endpoint(self):
#         return
#
#     @model_endpoint.setter
#     @abc.abstractmethod
#     def model_endpoint(self, model_endpoint: str):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def provider_key(self):
#         return
#
#     @provider_key.setter
#     @abc.abstractmethod
#     def provider_key(self, provider_key: str):
#         pass
#
#     @property
#     @abc.abstractmethod
#     def provider_type(self):
#         return
#
#     @provider_type.setter
#     @abc.abstractmethod
#     def provider_type(self, provider_key: str):
#         pass

class AiSuiteGoogleVertexEndpoint(AiSuiteModel):
    provider_type: ProviderType
    project_id: str
    location: str
    app_creds_path: str
    model: str
    model_endpoint: str
    provider_key: str = 'googlevertex'

class AiSuiteGoogleGenEndpoint(AiSuiteModel):
    provider_key: str = 'googlegenai'
    provider_type: ProviderType
    model_endpoint: str
    model: str
    api_key: str

class AiSuiteHuggingfaceEndpoint(AiSuiteModel):
    provider_key: str = 'huggingface'
    provider_type: ProviderType
    hf_token: str
    model_endpoint: str
    model: str

class HuggingfaceModelEndpoint(BaseModel):
    hf_model: str
    model_endpoint: str
    pipeline: dict[str, object]
    pipeline_kwargs: dict[str, object]

# will add the others ?
AiSuiteModelEndpoint = typing.Union[AiSuiteGoogleVertexEndpoint, AiSuiteGoogleGenEndpoint, AiSuiteHuggingfaceEndpoint]

class ModelType(enum.Enum):
    EMBEDDING = 0
    GENERATIVE_LANGUAGE = 1

class GeminiModelEndpoint(BaseModel):
    gemini_model: str
    model_endpoint: str
    api_key: str
    model_type: ModelType = ModelType.GENERATIVE_LANGUAGE


@configuration_properties(prefix_name='model_server')
class ModelServerConfigProps(ConfigurationProperties):
    port: int
    host: str
    hf_model_endpoint: typing.Optional[dict[str, HuggingfaceModelEndpoint]]
    gemini_model_endpoint: typing.Optional[dict[str, GeminiModelEndpoint]]
    ai_suite_model_endpoint: typing.Optional[dict[str, AiSuiteModelEndpoint]]