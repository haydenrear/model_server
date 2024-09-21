from typing import Optional

from drools_py.data.dataset.datasets import T, U
from drools_py.data.dataset.data_items import ValidationItem, TestItem, TrainingItem


class LambadaData(ValidationItem):
    def __init__(self, input_value: str, output_value: str):
        super().__init__(input_value, output_value)


class HuggingFaceTrainingItem(TrainingItem[T]):
    def __init__(self, input_value: T, label: str):
        super().__init__(input_value, label)


class HuggingFaceTestItem(TestItem[T]):
    def __init__(self, input_value: T, label: str):
        super().__init__(input_value, label)


class HuggingFaceValidationItem(ValidationItem[U]):
    def __init__(self, input_value: U, label: Optional[U] = None):
        super().__init__(input_value, label)
