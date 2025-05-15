import base64
import dataclasses

import numpy as np

from python_util.json_to_from.base_json import FromJsonClass
from python_util.numpy_utils.dtype_util import from_dtype_name


@dataclasses.dataclass(init=True, eq=True, unsafe_hash=True)
class LongKeyLongValue(FromJsonClass):
    key: int

    @classmethod
    def from_dict(cls, message):
        return LongKeyLongValue(message['key'])


@dataclasses.dataclass(init=True)
class VectorizedNdArrayMap(FromJsonClass):
    data_map: dict[LongKeyLongValue, np.ndarray]

    @classmethod
    def from_dict(cls, message):
        indices = {value['mapKey']['key']: cls.np_array_from_txt(value['mapValue'])
                   for value in message['indexIndex']}
        underlying = cls.np_array_from_txt(message['underlying'])
        long_map = {}
        for idx, v in indices.items():
            long_map[LongKeyLongValue(idx)] = underlying[v]

        return VectorizedNdArrayMap(long_map)

    @classmethod
    def np_array_from_txt(cls, input_data: dict):
        decoded = base64.standard_b64decode(input_data['data'])
        out = np.frombuffer(decoded, dtype=from_dtype_name(input_data['dtype']))
        out.reshape(input_data['shape'])
        return out
