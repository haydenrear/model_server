import abc
import dataclasses
import json
import typing

import numpy as np

from metadata_extractor.messaging.reflectable_media_component import ToJsonClass, FromJsonClass, \
    ReflectableMediaItem
from model_server.model_endpoint.model_endpoints import PytorchModelEndpoint


@dataclasses.dataclass(init=True)
class EmbeddingData(ToJsonClass):
    data: np.ndarray
    shape: list[int]
    version: float

    def toJSON(self) -> str:
        return json.dumps({
            'data': self.data.dumps(),
            'shape': self.shape,
            'version': self.version
        })


@dataclasses.dataclass(init=True)
class ToEmbedMessageWss(FromJsonClass):
    media_items: typing.List[ReflectableMediaItem]

    @classmethod
    def from_dict(cls, message: dict):
        return ToEmbedMessageWss([
            ReflectableMediaItem.from_dict(media_item)
            for media_item in message['mediaItems']
        ])


class PytorchModelEmbeddingEndpoint(PytorchModelEndpoint[ToEmbedMessageWss, EmbeddingData],
                                    abc.ABC):
    pass


class TokenizationModelEndpoint(PytorchModelEmbeddingEndpoint, abc.ABC):
    pass
