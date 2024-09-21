import injector

from financial_classifier.financial_news_classifier import FinanceClassifier
from metadata_extractor.delegating_metadata_extractor import DelegatingMetadataExtractor


class ExtractMetadataController:

    @injector.inject
    def __init__(self,financial_classifier: DelegatingMetadataExtractor):
        self.financial_classifier = financial_classifier