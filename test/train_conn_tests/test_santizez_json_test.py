import json
import unittest

from model_server.model_endpoint.retryable_model import RetryableModel


class TestSanitizeJson(unittest.TestCase):
    def test_sanitize_json(self):
        RetryableModel.parse_as_json('{}')

