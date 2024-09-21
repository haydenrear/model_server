import abc
import dataclasses
import json
import typing
from abc import ABCMeta, abstractmethod
from typing import Optional


class FromJsonClass(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def from_dict(cls, message: dict):
        pass

    @classmethod
    def fromJSON(cls, message: str):
        return cls.from_dict(json.loads(message))


class ToJsonClass(metaclass=ABCMeta):

    @abc.abstractmethod
    def toJSON(self) -> str:
        pass


class AssetIndex(FromJsonClass):
    def __init__(self, index: typing.Dict[str, typing.List[str]], version: float = 1.0):
        self.index = index
        self.version = version

    @classmethod
    def from_dict(cls, created: dict):
        return AssetIndex(
            created['index'],
            created['version']
        )

    @classmethod
    def fromJSON(cls, in_message: str):
        created = json.loads(in_message)
        if created:
            return cls.from_dict(created)


class MediaFileMetadata:
    def __init__(self, file_path: str, metadata_path: str, signature: bytearray, version: float = 1.0):
        self.signature = signature
        self.version = version
        self.metadata_path = metadata_path
        self.file_path = file_path


@dataclasses.dataclass(init=True)
class MediaUri(FromJsonClass):
    uri: str
    source_id: str
    offset: int

    @classmethod
    def from_dict(cls, created: dict):
        return MediaUri(
            created['uri'] if 'uri' in created.keys() else None,
            created['source_id'] if 'source_id' in created.keys() else None,
            cls.create_get_offset(created)
        )

    @classmethod
    def fromJSON(cls, in_message: str):
        created = json.loads(in_message)
        if created:
            return cls.from_dict(created)

    @classmethod
    def create_get_offset(cls, input_data: dict) -> typing.Optional[int]:
        if 'offset' in input_data.keys() and input_data['offset']:
            return int(input_data['offset'])


ReflectableMediaItemT = typing.ForwardRef("ReflectableMediaItem")


@dataclasses.dataclass(init=True)
class MimeType(FromJsonClass):
    type: str
    subtype: str
    parameters: dict
    wildcardSubtype: bool
    subtypeSuffix: str
    concrete: bool
    wildcardType: bool
    charset: str

    @classmethod
    def fromJSON(cls, message: str):
        return cls.from_dict(json.loads(message))

    @classmethod
    def from_dict(cls, created: dict):
        return MimeType(
            created['type'],
            created['subtype'],
            created['parameters'],
            created['wildcardSubtype'],
            created['subtypeSuffix'],
            created['concrete'],
            created['wildcardType'],
            created['charset']
        )


class ReflectableMediaItem(FromJsonClass):
    def __init__(self, metadata: Optional[MediaFileMetadata], buffers: Optional[bytearray], file_extension: str,
                 mime_type: MimeType, index: AssetIndex, media_uri: MediaUri,
                 content: typing.List[ReflectableMediaItemT]):
        self.content = content
        self.media_uri = media_uri
        self.file_extension = file_extension
        self.buffers = buffers
        self.metadata = metadata
        self.mime_type = mime_type
        self.index = index

    @classmethod
    def from_dict(cls, message: dict):
        inner = []
        if message['content'] is not None:
            for r in message['content']:
                inner.append(ReflectableMediaItem.from_dict(r))

        return ReflectableMediaItem(
            message['metadata'],
            message['buffers'],
            message['fileExtension']['com.hayden.shared.media.MediaType'],
            MimeType.from_dict(message['mimeType']),
            AssetIndex.from_dict(message['index']),
            MediaUri.from_dict(message['mediaUri']),
            inner
        )


class WebMediaDataIndiv:
    def __init__(self, date: int, content: typing.List[ReflectableMediaItem]):
        self.content = content
        self.date = date


class MediaComponent:
    def __init__(self, values: typing.List[ReflectableMediaItem], date: int, id: str, index: AssetIndex):
        self.index = index
        self.id = id
        self.date = date
        self.values = values


def get_key(key: str, loaded) -> Optional:
    if key in loaded.keys():
        return loaded[key]


def get_key_in(key: list[str], loaded) -> Optional:
    out = None
    for key_item in key:
        out = get_key(key_item, loaded if not out else out)
    return out


class ReflectableComponent:
    def __init__(self, components: dict[str, list[MediaComponent]], id: str, symbol: str):
        self.id = id
        self.symbol = symbol
        self.components = components


def serialize_media_component(message: str) -> ReflectableComponent:
    message = json.loads(message)
    reflectable_id = get_key_in(['data', 'WebData', 'inner', 'uniqueId'], message)
    value = get_key_in(['data', 'WebData', 'inner', 'data'], message)
    media_components = {}
    for key, items in value.items():
        media_components[key] = []
        for item in items:
            date = get_key('date', item)
            media_component_index = get_key('index', item)
            media_component_indices = get_key('index', media_component_index)
            media_component_version = get_key('version', media_component_index)
            values = get_key('values', item)
            indiv = []
            for value in values:
                indiv.append(ReflectableMediaItem.from_dict(value))

            media_components[key].append(
                MediaComponent(indiv, date, reflectable_id,
                               AssetIndex(media_component_indices, media_component_version))
            )

    return ReflectableComponent(media_components, reflectable_id, reflectable_id)


def deserialize_metadata(metadata: MediaFileMetadata):
    return {
        "filePath": metadata.file_path,
        "metadataPath": metadata.metadata_path,
        "version": metadata.version
    } if metadata else None


def deserialize_index(index):
    return {
        "index": index.index,
        "version": index.version
    }


def deserialize_media_component_indiv(indivs: dict[str, list[MediaComponent]]):
    return {
        key: [
            {
                "className": "com.hayden.reflectable.media.MediaData$WebMediaDataIndiv",
                "enumClzz": "com.hayden.shared.models.data.WebData",
                "values": [
                    {
                        "buffers": indiv.buffers,
                        "metadata": deserialize_metadata(indiv.metadata),
                        "fileExtension": indiv.file_extension,
                        "mimeType": indiv.mime_type,
                        "index": deserialize_index(indiv.index)
                    }
                    for indiv in media_component.values],
                "date": media_component.date,
                "index": deserialize_index(media_component.index)
            }
            for media_component in value] for key, value in indivs.items()
    }


def deserialize_media_component(message: ReflectableComponent) -> str:
    return json.dumps({
        "className": "com.hayden.shared.models.reflectable.ReflectableComponent",
        "data": {
            "WebData": {
                "className": "com.hayden.shared.models.reflectable.ReflectableWrapper",
                "inner": {
                    "className": "com.hayden.reflectable.media.MediaData",
                    "clzz": "com.hayden.asset.models.asset.MultiModalMediaFeed",
                    "symbol": message.id,
                    "enumClzz": "com.hayden.shared.models.data.WebData",
                    "reflectableClzz": "com.hayden.reflectable.media.MediaData$WebMediaDataIndiv",
                    "uniqueId": message.id,
                    "data": deserialize_media_component_indiv(message.components)
                },
                "type": "com.hayden.reflectable.media.ReflectableMediaItem"
            }
        }
    })
