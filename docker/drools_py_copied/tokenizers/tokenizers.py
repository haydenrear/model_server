import abc

import torch
from transformers import AutoTokenizer

from drools_py.configs.config_factory import ConfigFactory
from drools_py.configs.config_models import TokenizerId
from drools_py.configs.config import Config
from drools_py.data.constant import ATTENTION_MASK, INPUT_IDS
from transformers import AutoTokenizer


class FoundationTokenizerConfig(Config, abc.ABC):
    pass


class FoundationTokenizer(abc.ABC):
    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        pass

    @vocab_size.setter
    @abc.abstractmethod
    def vocab_size(self, vocab_size: int):
        pass

    @property
    @abc.abstractmethod
    def eos_token_id(self) -> int:
        pass

    @eos_token_id.setter
    @abc.abstractmethod
    def eos_token_id(self, eos_token_id: int):
        pass


class HuggingfaceTokenizerConfig(FoundationTokenizerConfig):

    def __init__(self, tokenizer_id: TokenizerId):
        self.tokenizer_id = tokenizer_id


class FoundationTokenizerFactory(ConfigFactory, abc.ABC):
    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def eos_token_id(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def pad_token_id(self) -> int:
        pass

    @abc.abstractmethod
    def special_tokens(self) -> list[int]:
        pass

    @abc.abstractmethod
    def get_token_id(self, t: str) -> int:
        pass


class TestFoundationTokenizerFactory(FoundationTokenizerFactory):
    def get_token_id(self, t: str) -> int:
        return 0

    def special_tokens(self) -> list[int]:
        return []

    def create(self, **kwargs):
        t = AutoTokenizer.from_pretrained('bert-base-uncased')
        return lambda x: t("hello there!", return_tensors='pt')

    @property
    def vocab_size(self) -> int:
        return -1

    @property
    def eos_token_id(self) -> int:
        return -1

    @property
    def pad_token_id(self) -> int:
        return -1


class HuggingFaceTokenizerConfigFactory(FoundationTokenizerFactory):

    def __init__(self, config_of_item_to_create: HuggingfaceTokenizerConfig):
        super().__init__(config_of_item_to_create)
        self.config_of_item_to_create = config_of_item_to_create
        tokenizer_name = self.config_of_item_to_create.tokenizer_id.config_option
        if 'gpt2' in tokenizer_name:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token_id
            self._vocab_size = self.tokenizer.vocab_size
            self._eos_token_id = self.tokenizer.eos_token_id
            self._pad_token_id = self.tokenizer.pad_token_id
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self._vocab_size = self.tokenizer.vocab_size
            self._pad_token_id = self.tokenizer.pad_token_id

    def create(self, **kwargs):
        # TODO: tokenizer kwargs.
        return self

    def __call__(self, t, **kwargs):
        return self.tokenizer(t, return_tensors='pt', truncation=True, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    def special_tokens(self) -> list[int]:
        from transformers import GPT2Tokenizer
        if isinstance(self.tokenizer, GPT2Tokenizer):
            return self.tokenizer.all_special_ids
        else:
            return []

    def get_token_id(self, t: str) -> int:
        ids_ = self.tokenizer(t, return_tensors='pt', truncation=True)[INPUT_IDS]
        return ids_[0]
