import abc
import enum
import os
import typing

from pydantic import ConfigDict
from pydantic.main import BaseModel

from aisuite.provider import ProviderType
from python_di.env.base_module_config_props import ConfigurationProperties
from python_di.properties.configuration_properties_decorator import configuration_properties

class AiSuiteModelType(enum.Enum):
    AiSuiteGoogleVertexEndpoint = "AiSuiteGoogleVertexEndpoint"
    AiSuiteGoogleGenEndpoint = "AiSuiteGoogleGenEndpoint"
    AiSuiteGoogleCloudDiscoveryEndpoint = "AiSuiteGoogleCloudDiscoveryEndpoint"
    AiSuiteHuggingfaceEndpoint = "AiSuiteHuggingfaceEndpoint"
    AiSuiteLangchainEndpoint = "AiSuiteLangchainEndpoint"


class AiSuiteModel(BaseModel, abc.ABC):
    model_config = ConfigDict(extra="allow")

class AiSuiteGoogleVertexEndpoint(AiSuiteModel):
    provider_type: ProviderType
    project_id: str
    location: str
    app_creds_path: str
    model: str
    model_endpoint: str
    provider_key: str = 'googlevertex'
    ai_suite_model_type: AiSuiteModelType = AiSuiteModelType.AiSuiteGoogleVertexEndpoint

class AiSuiteLangchainEndpoint(AiSuiteModel):
    provider_type: ProviderType
    model_endpoint: str
    model: typing.Optional[str] = None
    provider_key: str = 'langchain'
    ai_suite_model_type: AiSuiteModelType = AiSuiteModelType.AiSuiteLangchainEndpoint

class AiSuiteGoogleGenEndpoint(AiSuiteModel):
    provider_key: str = 'googlegenai'
    provider_type: ProviderType
    model_endpoint: str
    model: str
    api_key: str
    ai_suite_model_type: AiSuiteModelType = AiSuiteModelType.AiSuiteGoogleGenEndpoint

class AiSuiteGoogleCloudDiscoveryEndpoint(AiSuiteModel):
    provider_key: str = 'googlegenai'
    provider_type: ProviderType
    model_endpoint: str
    model: str
    application_credential: str
    project_id: str
    ai_suite_model_type: AiSuiteModelType = AiSuiteModelType.AiSuiteGoogleCloudDiscoveryEndpoint

class AiSuiteHuggingfaceEndpoint(AiSuiteModel):
    provider_key: str = 'huggingface'
    provider_type: ProviderType
    hf_token: str
    model_endpoint: str
    model: str
    ai_suite_model_type: AiSuiteModelType = AiSuiteModelType.AiSuiteHuggingfaceEndpoint

class HuggingfaceModelEndpoint(BaseModel):
    hf_model: str
    model_endpoint: str
    pipeline: dict[str, object]
    pipeline_kwargs: dict[str, object]


AiSuiteModelEndpoint = typing.Union[AiSuiteGoogleVertexEndpoint, AiSuiteGoogleGenEndpoint,
                                    AiSuiteHuggingfaceEndpoint, AiSuiteGoogleCloudDiscoveryEndpoint, AiSuiteLangchainEndpoint]

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
    hf_model_endpoint: typing.Optional[dict[str, HuggingfaceModelEndpoint]] = None
    gemini_model_endpoint: typing.Optional[dict[str, GeminiModelEndpoint]] = None
    ai_suite_model_endpoint: typing.Optional[dict[str, AiSuiteModelEndpoint]] = None