import abc
import enum

import torch

from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory
from drools_py.serialize.serializable_types import SerializableEnum


class SimilarityType(SerializableEnum):
    Cosine = enum.auto()


class SimilarityConfig(Config):

    def __init__(self, similarity_type: SimilarityType):
        self.similarity_type = similarity_type

    @staticmethod
    def test_properties(**kwargs) -> dict:
        return SimilarityConfig.update_override(kwargs,
                                                SimilarityConfig(SimilarityType.Cosine).to_self_dictionary())


class SimilarityConfigFactory(ConfigFactory):


    def create(self, **kwargs):
        assert isinstance(self.config_of_item_to_create, SimilarityConfig)
        if self.config_of_item_to_create.similarity_type == SimilarityType.Cosine:
            return torch.nn.CosineSimilarity(**kwargs)

    @staticmethod
    def test_properties(**kwargs) -> dict:
        return SimilarityConfigFactory.update_override(
            kwargs,
            SimilarityConfigFactory(SimilarityConfig.build_test_config()).to_self_dictionary()
        )