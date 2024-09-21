import abc
from typing import Optional

from drools_py.classification_models.torch_classification import DistilBertStringClassifier, StringClassifier


class BbcNewsClassifier(DistilBertStringClassifier):

    def __init__(self, classification_threshold: Optional[float] = None):
        super().__init__('Umesh/distilbert-bbc-news-classification',
                         classification_threshold=classification_threshold,
                         classifications={0: "entertainment", 1: "politics", 2: "tech", 3: "sport", 4: "business"})


class NewsCategoryClassifier(DistilBertStringClassifier):

    def __init__(self, classification_threshold: Optional[float] = None):
        self.classifications = {0: 'sports', 1: 'news_&_social_concern', 2: 'fitness_&_health',
                                3: 'youth_&_student_life', 4: 'learning_&_educational', 5: 'science_&_technology',
                                6: 'celebrity_&_pop_culture', 7: 'travel_&_adventure', 8: 'diaries_&_daily_life',
                                9: 'food_&_dining', 10: 'gaming', 11: 'business_&_entrepreneurs', 12: 'family',
                                13: 'relationships', 14: 'fashion_&_style', 15: 'music', 16: 'film_tv_&_video',
                                17: 'other_hobbies', 18: 'arts_&_culture'}

        super().__init__('cardiffnlp/tweet-topic-21-multi',
                         classification_threshold=classification_threshold,
                         classifications=self.classifications)
