import dataclasses

import injector

from drools_py.batch_split.split_batch_emit import SplitBatchEmit
from drools_py.batch_split.split_seq_emit import SplitSeqEmit
from drools_py.data.constant import ATTENTION_MASK, INPUT_IDS, DECORATED_DATA_METADATA, OUT_OF, N
from drools_py.data.dataset.dataset_decorator.get_item_decorator import GetItemDecorator
from drools_py.data.dataset.dataset_decorator.tokenization_decorator import TokenizationDecorator
from python_di.configs.component import component
from python_util.logger.logger import LoggerFacade


@dataclasses.dataclass(init=True)
class DecoratedDataMetadata:
    n: int
    total: int

    def to_dict(self) -> dict:
        return {N: self.n, OUT_OF: self.total}


@component(bind_to=[GetItemDecorator])
class SplitEmitDecorator(GetItemDecorator):

    @injector.inject
    def __init__(self, split_seq: SplitSeqEmit, split_batch: SplitBatchEmit):
        self.split_seq = split_seq
        self.split_batch = split_batch
        self.q = []

    def get_dependencies(self) -> list[type]:
        from drools_py.data.dataset.data_items.strategies import DataLoadStrategy
        return [TokenizationDecorator, DataLoadStrategy]

    def get_item(self, idx=None) -> ...:
        if idx is None:
            return self._do_pop_if_poppable()

        attn_mask, inputs = self.assert_process_inputs(idx)

        # this does not have a state, as it returns with the attention mask
        seq_split = self.split_seq.do_register_emit(inputs, None, attn_mask, None)

        split_batches = []
        for prev_state, encoder_state, src_mask, tgt_mask in seq_split:
            assert encoder_state is None
            batch_split = self.split_batch.do_register_emit(prev_state, encoder_state, src_mask, tgt_mask)
            split_batches.extend(batch_split)

        total = len(split_batches)
        for j, (prev_state_, encoder_state_, src_mask_, tgt_mask_) in enumerate(split_batches):
            assert encoder_state_ is None
            next_value = {ATTENTION_MASK: tgt_mask_,
                          INPUT_IDS: self._post_process_input(prev_state_),
                          DECORATED_DATA_METADATA: DecoratedDataMetadata(j + 1, total)}

            self.q.append(next_value)

        return self._do_pop_if_poppable()

    def assert_process_inputs(self, idx):
        idx = self._process_data(idx)
        attn_mask, inputs = self._assert_get_attn_mask_inputs(idx)
        inputs = self._pre_process_input(inputs)
        return attn_mask, inputs

    @staticmethod
    def _post_process_input(inputs):
        return inputs.transpose(0, 1).squeeze(2)

    @staticmethod
    def _pre_process_input(inputs):
        if len(inputs.shape) == 2:
            inputs = inputs.T
            inputs = inputs.unsqueeze(2)
            assert inputs.shape[-1] == 1

        return inputs

    @staticmethod
    def _assert_get_attn_mask_inputs(idx):
        assert isinstance(idx, dict), "Input to split emit decorator must be output from tokenizer."
        assert ATTENTION_MASK in idx.keys()
        assert INPUT_IDS in idx.keys()
        attn_mask = idx[ATTENTION_MASK]
        inputs = idx[INPUT_IDS]
        return attn_mask, inputs

    def _process_data(self, idx):
        if not isinstance(idx, dict | None):
            assert hasattr(idx, 'data'), f"{idx} does not have property data."
            idx = idx.data
        return idx

    def _do_pop_if_poppable(self):
        return self.q.pop(0) if len(self.q) != 0 else None

    def has_item(self) -> bool:
        return len(self.q) != 0
