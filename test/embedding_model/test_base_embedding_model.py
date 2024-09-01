import json
import os
from unittest import TestCase

from model_server.embedding_model.base_embedding_model import ToEmbedMessageWss
from python_util.io_utils.file_dirs import get_dir


class TestToEmbedMessageWss(TestCase):
    def test_serialize_reflectable_media_item(self):
        resources = get_dir(__file__, 'test_work')
        json_file = os.path.join(resources, 'embedded_msg.json')
        assert os.path.exists(json_file)
        with open(json_file, 'r') as f:
            loaded = json.load(f)
            to_embed = ToEmbedMessageWss.from_dict(loaded)
            assert len(to_embed.media_items) == 1
            assert len(to_embed.media_items[0].content) == 1
