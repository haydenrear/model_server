import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SpamClassifier:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("johnpaulbin/autotrain-spam-39547103148")
        self.tokenizer = AutoTokenizer.from_pretrained("johnpaulbin/autotrain-spam-39547103148")
        self.softmax = torch.nn.Softmax()

    def return_value(self, to_classify: str) -> torch.Tensor:
        inputs = self.tokenizer(to_classify, return_tensors="pt")

        model_out: torch.Tensor = self.model(**inputs)
        return self.softmax(model_out[0][0])

    def classify_uri_is_spam(self, to_classify: str) -> bool:
        after = self.split_and_get_after(to_classify)
        updated = ' '.join(after.split('-'))
        updated = updated.split('/')[0]
        updated = updated.split('.')[0]
        outputs = self.return_value(updated)
        print(outputs)

        return outputs[0] < 0.1

    def classify_text_as_spam(self, to_classify: str) -> torch.Tensor:
        return self.return_value(to_classify)

    def split_and_get_after(self, to_classify):
        split = to_classify.split('https://')
        if len(split) > 1:
            to_classify = split[1]
        else:
            to_classify = split[0]
        return to_classify
