import base64
import codecs
import json
import os
from unittest import TestCase

import numpy as np

from model_server.generative_model.vectorized_map import LongKeyLongValue, VectorizedNdArrayMap
from python_util.io_utils.file_dirs import get_dir
from python_util.numpy_utils.dtype_util import from_dtype_name


class TestVectorizedNdArrayMap(TestCase):
    def test_from_dict(self):
        opened = get_dir(__file__, 'test_work')
        out_json = os.path.join(opened, 'out.json')
        assert os.path.exists(out_json)
        with open(out_json, 'r') as o:
            loaded = json.load(o)
            created = VectorizedNdArrayMap.from_dict(loaded)

            assert len(created.data_map) != 0
            assert len(created.data_map) == 167

            for i in range(100):
                assert created.data_map[LongKeyLongValue(i)].tolist() == [i]
            for i in range(100, 167):
                assert created.data_map[LongKeyLongValue(i)].tolist() == [10]

    def test_load(self):
        opened = get_dir(__file__, 'test_work')
        with open(os.path.join(opened, 'np_array.json'), 'r') as o:
            created = json.load(o)
            decoded = base64.standard_b64decode(created['data'])
            out = np.frombuffer(decoded, dtype=from_dtype_name(created['dtype']))
            out.reshape(created['shape'])
            assert out[0] == 1
            assert out[1] == 2
            assert out[2] == 5
            assert len(out) == 3
