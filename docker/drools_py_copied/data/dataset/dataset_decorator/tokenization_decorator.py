import injector

from drools_py.data.dataset.data_items.data_items import TextTrainingItem
from drools_py.data.dataset.dataset_decorator.get_item_decorator import GetItemDecorator
from drools_py.tokenizers.tokenizers import FoundationTokenizerFactory
from python_di.configs.component import component


@component(bind_to=[GetItemDecorator])
class TokenizationDecorator(GetItemDecorator):

    @injector.inject
    def __init__(self, tokenizer: FoundationTokenizerFactory):
        self.tokenizer = tokenizer.create()

    def get_dependencies(self) -> list[type]:
        from drools_py.data.dataset.data_items.strategies import DataLoadStrategy
        return [DataLoadStrategy]

    def get_item(self, idx) -> ...:
        if isinstance(idx, TextTrainingItem ):
            return self.tokenizer(idx.src)
        elif isinstance(idx, str):
            return self.tokenizer(idx)

    def has_item(self) -> bool:
        return False
