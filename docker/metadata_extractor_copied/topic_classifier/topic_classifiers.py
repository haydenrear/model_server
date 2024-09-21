from typing import Optional

from drools_py.classification_models.torch_classification import DistilBertStringClassifier


class WebSiteClassifier(DistilBertStringClassifier):
    def __init__(self, classification_threshold: Optional[float]=None):
        super().__init__('alimazhar-110/website_classification',
                         'alimazhar-110/website_classification',
                         classification_threshold=classification_threshold,
                         classifications={0: 'social networking and messaging', 1: 'adult', 2: 'photography',
                                          3: 'e-commerce', 4: 'streaming services', 5: 'computers and technology',
                                          6: 'health and fitness', 7: 'sports', 8: 'food', 9: 'news',
                                          10: 'law and government', 11: 'games', 12: 'business/corporate', 13: 'travel',
                                          14: 'education', 15: 'forums'})


class TopicClassification(DistilBertStringClassifier):

    def __init__(self, classification_threshold: Optional[float]=None):
        super().__init__('ebrigham/EYY-Topic-Classification',
                         classification_threshold=classification_threshold,
                         classifications={0: 'participation and engagement', 1: 'n/a', 2: 'culture',
                                          3: 'democratic values', 4: 'policy dialogues', 5: 'digital',
                                          6: 'natural sustainability', 7: 'climate change', 8: 'health and well-being',
                                          9: 'research and innovation', 10: 'youth and the world',
                                          11: 'european learning mobility', 12: 'renewable energy',
                                          13: 'employment and inclusion', 14: 'studying abroad', 15: 'education'})


class ArkViClassifier(DistilBertStringClassifier):
    def __init__(self, classification_threshold: Optional[float]=None):
        self.classifications = {0: 'economics', 1: 'mathematics', 2: 'statistics', 3: 'quantitative finance',
                                4: 'computer science', 5: 'electrical engineering and systems science', 6: 'physics',
                                7: 'mathematical physics', 8: 'nonlinear sciences', 9: 'quantitative biology',
                                10: 'high energy physics - theory', 11: 'general relativity and quantum cosmology',
                                12: 'high energy physics - phenomenology', 13: 'quantum physics',
                                14: 'condensed matter', 15: 'astrophysics', 16: 'high energy physics - lattice',
                                17: 'high energy physics - experiment', 18: 'nuclear experiment', 19: 'nuclear theory',
                                20: 'other'}

        super().__init__('Wi/arxiv-topics-distilbert-base-cased',
                         'Wi/arxiv-topics-distilbert-base-cased',
                         classification_threshold=classification_threshold,
                         classifications=self.classifications)
