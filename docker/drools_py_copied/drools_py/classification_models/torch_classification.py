import abc
from typing import Optional

import torch.nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def get_base_tokenizer(tokenizer):
    return tokenizer if tokenizer else 'bert-base-uncased'


class TorchClassificationModel(abc.ABC):

    @property
    @abc.abstractmethod
    def classifications(self) -> dict[int, str]:
        pass

    @classifications.setter
    @abc.abstractmethod
    def classifications(self, classifications):
        pass

    @property
    @abc.abstractmethod
    def classification_threshold(self) -> float:
        pass

    @classification_threshold.setter
    @abc.abstractmethod
    def classification_threshold(self, classification_threshold: float):
        pass

    def classification_or_none(self, probabilities: torch.Tensor):
        max_probability, predicted_class = torch.max(probabilities, dim=-1)
        print(f'{probabilities} are proba, {self.classification_threshold} is threshold, and {max_probability} is max')
        if max_probability.item() > self.classification_threshold:
            return self.classifications[int(predicted_class)]
        else:
            return None

    @property
    @abc.abstractmethod
    def tokenizer(self):
        pass

    @tokenizer.setter
    @abc.abstractmethod
    def tokenizer(self, tokenizer):
        pass

    @abc.abstractmethod
    def get_probabilities(self, in_values: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def model(self):
        pass

    @model.setter
    @abc.abstractmethod
    def model(self, model):
        pass


class StringClassifier(abc.ABC):
    @abc.abstractmethod
    def classify_item(self, item: str) -> Optional[str]:
        pass


class TorchStringClassifier(StringClassifier, TorchClassificationModel, abc.ABC):

    def __init__(self, threshold: Optional[float] = None, properties: dict[int, str] = {}):
        StringClassifier.__init__(self)
        TorchClassificationModel.__init__(self)
        self._classifications = properties
        if threshold is None:
            if len(self._classifications) == 0:
                self._classification_threshold = 0.5
                print(f'set {self._classification_threshold} for {self._classifications}')
            else:
                self._classification_threshold = (len(self._classifications) // 4) / len(self._classifications)
                print(f'set {self._classification_threshold} for {self._classifications}')
        else:
            self._classification_threshold = threshold

    @property
    def classifications(self) -> dict[int, str]:
        return self._classifications

    @classifications.setter
    def classifications(self, threshold: dict[int, str]):
        self._classifications = threshold

    @property
    def classification_threshold(self) -> float:
        return self._classification_threshold

    @classification_threshold.setter
    def classification_threshold(self, threshold: float):
        self._classification_threshold = threshold

    def classify_item(self, item: str) -> Optional[str]:
        tokenized = self.tokenizer(item, return_tensors='pt')
        return self.classification_or_none(
            self.get_probabilities(self.model(**tokenized)[0][0])
        )


class HasTorchProperties(TorchClassificationModel, abc.ABC):
    def __init__(self, tokenizer, model):
        TorchClassificationModel.__init__(self)
        self._tokenizer = None
        self._model = None
        self.tokenizer = tokenizer
        self.model = model

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model


class DistilBertStringClassifier(TorchStringClassifier, HasTorchProperties, abc.ABC):
    def __init__(self, model_name: str,
                 tokenizer: Optional[str] = None,
                 classification_threshold: Optional[float] = None,
                 classifications: dict[int, str] = {}):
        HasTorchProperties.__init__(self,
                                    DistilBertTokenizerFast.from_pretrained(get_base_tokenizer(tokenizer)),
                                    DistilBertForSequenceClassification.from_pretrained(model_name))
        TorchStringClassifier.__init__(self, classification_threshold, classifications)
        self._classifications = classifications
        self.softmax = torch.nn.Softmax()

    def get_probabilities(self, in_values: torch.Tensor) -> torch.Tensor:
        return self.softmax(in_values)


class BertForSequenceClassificationStringClassifier(TorchStringClassifier, HasTorchProperties, abc.ABC):
    def __init__(self,
                 model_name: str,
                 tokenizer: Optional[str] = None,
                 classification_threshold: Optional[float] = None,
                 classifications: dict[int, str] = {}
                 ):
        self._classifications = None
        self._classification_threshold = None
        self._tokenizer = None
        self._model = None
        HasTorchProperties.__init__(self, BertTokenizer.from_pretrained(get_base_tokenizer(tokenizer)),
                                    BertForSequenceClassification.from_pretrained(model_name))
        TorchStringClassifier.__init__(self, classification_threshold, classifications)
        self.softmax = torch.nn.Softmax()

    def get_probabilities(self, in_values: torch.Tensor) -> torch.Tensor:
        return self.softmax(in_values)


class AutoModelStringClassifier(TorchStringClassifier, HasTorchProperties, abc.ABC):
    def __init__(self,
                 model_name: str,
                 tokenizer: Optional[str] = None,
                 classification_threshold: Optional[float] = None,
                 classifications: dict[int, str] = {}):
        self._classifications = None
        self._classification_threshold = None
        self._tokenizer = None
        self._model = None
        HasTorchProperties.__init__(self,AutoTokenizer.from_pretrained(get_base_tokenizer(tokenizer)),
                                    AutoModelForSequenceClassification.from_pretrained(model_name)
                                    )
        TorchStringClassifier.__init__(self, classification_threshold, classifications)
        self.softmax = torch.nn.Softmax()

    def get_probabilities(self, in_values: torch.Tensor) -> torch.Tensor:
        return self.softmax(in_values)
