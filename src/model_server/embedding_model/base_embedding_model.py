import dataclasses
import json

import numpy as np

from python_util.json_to_from.base_json import ToJsonClass


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


