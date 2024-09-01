import typing

import injector
from fontTools.misc.plistlib import end_real
from pasta.base.codegen_test import AutoFormatTest

from model_server.model_endpoint.model_endpoints import ModelEndpoint
from model_server.train_conn.server_config_props import ModelServerConfigProps, HuggingfaceModelEndpoint
from python_di.configs.component import component
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
import transformers
import torch

from python_di.configs.prototype import prototype_scope_bean, prototype_factory


# model = "codellama/CodeLlama-7b-hf"
#
# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "feature-extraction",
#     model=model
# )

# sequences = pipeline(
#     'import socket\n\ndef ping_exponential_backoff(host: str):',
#     do_sample=True,
#     top_k=10,
#     temperature=0.1,
#     top_p=0.95,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=200,
# )


@prototype_scope_bean(bindings=[ModelEndpoint])
class HfEndpoint(ModelEndpoint):

    @prototype_factory()
    def __init__(self, model_server_props: ModelServerConfigProps):
        self.model_server_props = model_server_props
        self._hf: typing.Optional[HuggingfaceModelEndpoint] = None
        self.pipeline = None
        self.tokenizer = None

    @property
    def endpoint(self) -> str:
        return self._hf.model_endpoint

    @property
    def hf(self) -> typing.Optional[HuggingfaceModelEndpoint]:
        return self._hf

    @hf.setter
    def hf(self, hf: typing.Optional[HuggingfaceModelEndpoint]):
        self._hf = hf
        self.tokenizer = AutoTokenizer.from_pretrained(self._hf.pipeline['model'])
        self.pipeline = transformers.pipeline(
            task=self._hf.pipeline['task'],
            model=self._hf.pipeline['model']
        )

    def do_model(self, input_data: dict[str, str]):
        # TODO: pooling type - PCA? VAE? t-SNE?
        return {
            'data': torch.mean(self.pipeline(input_data['prompt'], **self._hf.pipeline_kwargs, return_tensors='pt'))[0]
        }
