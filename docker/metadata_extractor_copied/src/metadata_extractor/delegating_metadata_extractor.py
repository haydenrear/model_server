import abc

import injector
from drools_py.classification_models.classify_spam import SpamClassifier
from drools_py.classification_models.torch_classification import StringClassifier
from python_di.inject.injector_provider import InjectionContext


class MultiStringClassifier(abc.ABC):

    @abc.abstractmethod
    def classify_item(self, item: str) -> list[str]:
        pass


class DelegatingMetadataExtractor(MultiStringClassifier):

    @injector.inject
    def __init__(self,
                 classifiers: list[StringClassifier],
                 spam_classifier: SpamClassifier):
        self.spam_classifier = spam_classifier
        self.classifiers = classifiers
        threshold = InjectionContext.environment.get_property('spam_classification_threshold')
        if threshold and isinstance(threshold, float):
            print(f"Threshold: {threshold} was a float")
            self.spam_classification_threshold = threshold
        else:
            print(f"Threshold: {threshold} was not a float")
            self.spam_classification_threshold = 0.5


    def classify_item(self, item: str) -> list[str]:
        classifications = []

        spam_classification = self.spam_classifier.classify_text_as_spam(item)
        if spam_classification[0] > self.spam_classification_threshold:
            print(f"Skipped itm with spam classification: {spam_classification[0]}")
            return classifications

        for classify in self.classifiers:
            item = classify.classify_item(item)
            if item is not None:
                classifications.append(item)
        return classifications
