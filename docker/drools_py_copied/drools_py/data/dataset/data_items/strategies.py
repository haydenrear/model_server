import enum
import typing

import injector

from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory
from drools_py.configs.config_models import EnumConfigOption, ConfigOption
from drools_py.data.dataset.data_items.data_item_factory import DataItemFactory, DataItemConfigFactory
from drools_py.data.dataset.data_items.data_items import DataItem
from drools_py.data.dataset.dataset_decorator.get_item_decorator import GetItemDecorator
from drools_py.data.dataset.datasets.abstract_datasets import DataType, AbstractLoadStrategy
from python_di.configs.component import component
from python_di.inject.profile_composite_injector.composite_injector import profile_scope
from python_util.collections.collection_util import first_key


def get_data_item_gpt_4_tiny_stories(next_training_item):
    prompt = next_training_item['prompt']
    story = next_training_item['story']
    return prompt, story


def get_data_item_src_tgt(next_training_item):
    next_key = first_key(next_training_item)
    return next_key, next_training_item[next_key]


def get_data_item_same_sequence(next_training_item):
    next_key = first_key(next_training_item)
    tgt = next_training_item[next_key]
    return tgt, tgt


def get_data_item_txt_src_data_load(next_training_item):
    assert isinstance(next_training_item, dict) and 'text' in next_training_item.keys(), \
        f"{next_training_item} failed assertion that had text."
    prompt = next_training_item['text']
    story = next_training_item['text']
    return prompt, story


def get_text_src_data_load(next_training_item):
    return next_training_item['text']


class DataLoadStrategyOption(EnumConfigOption):
    SameSequenceDataItem = enum.auto()
    TinyStoriesGpt4 = enum.auto()
    SrcTgtDataLoadStrategy = enum.auto()
    TextSrcDataLoadStrategy = enum.auto()
    TextSrcLoadStrategy = enum.auto()


class DataLoadStrategyConfigOption(ConfigOption[DataLoadStrategyOption]):

    def __init__(self, config_option: DataLoadStrategyOption = DataLoadStrategyOption.SrcTgtDataLoadStrategy):
        super().__init__(config_option)


class DataLoadStrategyConfig(Config):
    def __init__(self,
                 load_strategy: DataLoadStrategyConfigOption,
                 data_item_factory: DataItemConfigFactory):
        self.data_item_factory = data_item_factory
        self.load_strategy = load_strategy


class DataLoadStrategy(GetItemDecorator, AbstractLoadStrategy):

    def __init__(self, config: DataLoadStrategyConfig, data_load: DataType):
        self.data_item_factory = config.data_item_factory.create(data_load)
        self.load_strategy = self._get_create_factory(config.load_strategy.config_option)

    def get_dependencies(self) -> list[type]:
        return []

    def get_item(self, next_training_item):
        loaded = self.load_strategy(next_training_item)
        return self.data_item_factory.produce(loaded)

    @staticmethod
    def _get_create_factory(value: DataLoadStrategyOption):
        if value == DataLoadStrategyOption.SameSequenceDataItem:
            return get_data_item_same_sequence
        elif value == DataLoadStrategyOption.TinyStoriesGpt4:
            return get_data_item_gpt_4_tiny_stories
        elif value == DataLoadStrategyOption.SrcTgtDataLoadStrategy:
            return get_data_item_src_tgt
        elif value == DataLoadStrategyOption.TextSrcDataLoadStrategy:
            return get_data_item_txt_src_data_load
        elif value == DataLoadStrategyOption.TextSrcLoadStrategy:
            return get_text_src_data_load


