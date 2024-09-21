from typing import Optional

from drools_py.classification_models.torch_classification import AutoModelStringClassifier, \
    BertForSequenceClassificationStringClassifier, StringClassifier


class EconomicsClassifier(AutoModelStringClassifier):

    def __init__(self, classification_threshold: Optional[float]=None):
        self.classifications = {0: 'Economics', 1: 'Other'}

        super().__init__(model_name='hakonmh/topic-xdistil-uncased',
                         tokenizer="hakonmh/topic-xdistil-uncased",
                         classification_threshold=classification_threshold,
                         classifications=self.classifications)


class FinancialNewsClassifier(BertForSequenceClassificationStringClassifier):
    def __init__(self, classification_threshold: Optional[float]=None):
        self.classifications = {0: 'financials', 1: 'company | product news', 2: 'stock movement', 3: 'macro',
                                4: 'analyst update', 5: 'general news | opinion', 6: 'currencies', 7: 'earnings',
                                8: 'energy | oil', 9: 'fed | central banks', 10: 'm&a | investments',
                                11: 'stock commentary', 12: 'gold | metals | materials', 13: 'politics', 14: 'ipo',
                                15: 'legal | regulation', 16: 'personnel change', 17: 'dividend',
                                18: 'treasuries | corporate debt', 19: 'markets'}

        super().__init__('nickmuchi/finbert-tone-finetuned-finance-topic-classification',
                         'nickmuchi/finbert-tone-finetuned-finance-topic-classification',
                         classification_threshold=classification_threshold,
                         classifications=self.classifications)


class FinanceClassifier(StringClassifier):

    def __init__(self, classification_threshold: Optional[float]=None):
        self.econ = EconomicsClassifier()
        self.financial_news = FinancialNewsClassifier(classification_threshold)

    def classify_item(self, item: str) -> Optional[str]:
        if self.econ.classify_item(item) == 'Economics':
            return self.financial_news.classify_item(item)
