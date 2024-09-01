import abc
import dataclasses

import torch.nn

from metadata_extractor.messaging.reflectable_media_component import FromJsonClass, AssetIndex
from model_server.generative_model.vectorized_map import VectorizedNdArrayMap
from model_server.model_endpoint.model_endpoints import ModelEndpoint, OutT, PytorchModelEndpoint


class GenerativeModel(ModelEndpoint, abc.ABC):
    pass


@dataclasses.dataclass(init=True)
class DownstreamModelInput(FromJsonClass):
    inputs: VectorizedNdArrayMap
    index: AssetIndex

    @classmethod
    def from_dict(cls, message: dict):
        return DownstreamModelInput(
            VectorizedNdArrayMap.from_dict(message['inputs']),
            AssetIndex.from_dict(message['indices'])
        )


class PytorchGenerativeModel(PytorchModelEndpoint[DownstreamModelInput, OutT],
                             torch.nn.Module,
                             abc.ABC):
    pass


