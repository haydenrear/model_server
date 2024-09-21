import abc
import dataclasses
import typing
from typing import Optional

import torch.nn

from drools_py.configs.config import Config
from drools_py.configs.config_models import LayerNormFirst, TransparentAttentionEnabled, TransparentAttentionLayerNorm, \
    TransparentAttentionLayerNormConfigOption, EmbeddingSize, NumLayers
from drools_py.torch_utils.torch_prov_mod_configs import LayerNormConfig
from drools_py.weight_init.weight_initialization_config_types import WeightInitializationTypesLnConfigOption, \
    WeightInitializationTypesLnBiasConfigOption
from python_util.logger.logger import LoggerFacade
from python_util.torch_utils.pytorch_util import detach_tensor_state


class LayerNormDelegateConfig(Config):
    pass


class LayerNormMetadata(abc.ABC):
    pass


@dataclasses.dataclass
class MultiLayerLayerNormMetadata(LayerNormMetadata):
    idx: int
    layer_idx: int = 0
    agg: bool = True

    @classmethod
    def idx(cls, i: int):
        return MultiLayerLayerNormMetadata(i, 0)

    @classmethod
    def with_layer_idx(cls, i: int, layer_idx: int):
        return MultiLayerLayerNormMetadata(i, layer_idx)


@dataclasses.dataclass
class SingleLayerLayerNormMetadata(LayerNormMetadata):
    pass


class LayerNormDelegate(torch.nn.Module, abc.ABC):

    @abc.abstractmethod
    def start(self, layer_norm_metadata: LayerNormMetadata):
        pass

    @abc.abstractmethod
    def finish(self, layer_norm_metadata: LayerNormMetadata):
        pass

    @abc.abstractmethod
    def pre_ln(self, to_ln: torch.Tensor,
               layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None):
        pass

    @abc.abstractmethod
    def post_ln(self, to_ln: torch.Tensor,
                layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None):
        pass

    @abc.abstractmethod
    def pre_ta_ln(self, to_ln: torch.Tensor,
                  layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None,
                  agg: Optional[torch.Tensor] = None):
        pass

    @abc.abstractmethod
    def post_ta_ln(self, to_ln: torch.Tensor,
                   layer_norm_metadata: MultiLayerLayerNormMetadata,
                   agg: Optional[torch.Tensor] = None):
        """
        This will keep track of
        :param layer_norm_metadata:
        :param agg:
        :param to_ln:
        :return:
        """
        pass


class TorchSingleLayerNormDelegateConfig(LayerNormDelegateConfig):
    def __init__(self, ln: LayerNormConfig, ln_first: LayerNormFirst,
                 weight_init: WeightInitializationTypesLnConfigOption,
                 bias_init: WeightInitializationTypesLnBiasConfigOption):
        self.ln_first = ln_first
        self.ln = ln

    @classmethod
    def get_ln_config(
            cls,
            embed_dim: EmbeddingSize,
            layer_norm_encoder_decoder: LayerNormFirst,
            weight_init: WeightInitializationTypesLnConfigOption,
            bias_init: WeightInitializationTypesLnBiasConfigOption,
    ):
        return TorchSingleLayerNormDelegateConfig(
            LayerNormConfig.layer_norm_config_dim_override(dim=embed_dim, ),
            layer_norm_encoder_decoder, weight_init, bias_init
        )


class TorchSingleLayerNormDelegate(LayerNormDelegate):

    def __init__(self, layer_norm: TorchSingleLayerNormDelegateConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_norm = layer_norm.ln.to_layer_norm()
        self.ln_first = layer_norm.ln_first.config_option

    def start(self, layer_norm_metadata: LayerNormMetadata):
        pass

    def finish(self, layer_norm_metadata: LayerNormMetadata):
        pass

    def pre_ln(self, to_ln: torch.Tensor,
               layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None):
        if self.ln_first:
            to_ln = self.layer_norm(to_ln)
        return to_ln

    def post_ln(self, to_ln: torch.Tensor,
                layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None):
        if not self.ln_first:
            to_ln = self.layer_norm(to_ln)

        return to_ln

    def pre_ta_ln(self, to_ln: torch.Tensor,
                  layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None,
                  agg: Optional[torch.Tensor] = None):
        raise ValueError("Single layer norm cannot do transparent attention.")

    def post_ta_ln(self, to_ln: torch.Tensor,
                   layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None,
                   agg: Optional[torch.Tensor] = None):
        raise ValueError("Single layer norm cannot do transparent attention.")


class TorchMultiLayerNormDelegateConfig(LayerNormDelegateConfig):
    def __init__(self, ln: list[LayerNormConfig],
                 ta_enabled: TransparentAttentionEnabled,
                 ta_ln: TransparentAttentionLayerNormConfigOption,
                 ln_first: LayerNormFirst):
        self.ta_ln = ta_ln
        self.ln_first = ln_first
        self.ta_enabled = ta_enabled
        self.ln = ln

    @classmethod
    def get_multi_ln_config(
            cls,
            embed_dim: EmbeddingSize,
            ta_enabled: TransparentAttentionEnabled,
            num_decoding_sub_layer: NumLayers,
            encoder_decoder_transparent_ln: TransparentAttentionLayerNormConfigOption,
            layer_norm_encoder_decoder: LayerNormFirst
    ):
        return TorchMultiLayerNormDelegateConfig(
            [LayerNormConfig.layer_norm_config_dim_override(dim=embed_dim) for _
             in range(num_decoding_sub_layer.config_option)],
            ta_enabled,
            encoder_decoder_transparent_ln,
            layer_norm_encoder_decoder,
        )


class TorchMultiLayerNormDelegate(LayerNormDelegate):

    def __init__(self, layer_norm: TorchMultiLayerNormDelegateConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = layer_norm
        self.ta_ln = layer_norm.ta_ln
        self.ln = torch.nn.ModuleDict({str(i): ln.to_layer_norm() for i, ln in enumerate(layer_norm.ln)})
        self.ta_enabled = layer_norm.ta_enabled.config_option
        self.ln_first = layer_norm.ln_first.config_option

        self.idx_state = 0

        self.state = None

    def start(self, layer_norm_metadata: MultiLayerLayerNormMetadata):
        """
        Called when the layer norm for a particular item in the iterable is started, with the index.
        :param layer_norm_metadata:
        :return:
        """
        assert self.idx_state == 0 or layer_norm_metadata.idx == self.idx_state + 1, \
            f"Idx state {self.idx_state} was incorrect for {layer_norm_metadata.idx}."
        self.idx_state = layer_norm_metadata.idx

    def finish(self, layer_norm_metadata: MultiLayerLayerNormMetadata):
        if self.idx_state == len(self.ln) - 1:
            self.idx_state = 0
            self.state = None
            LoggerFacade.info("Finished layer normalization.")

    def pre_ln(self, to_ln: torch.Tensor,
               layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None):
        if self.ta_enabled:
            return to_ln
        if self.ln_first:
            to_ln = self.ln[str(self.idx_state)](to_ln)
        return to_ln

    def post_ln(self, to_ln: torch.Tensor,
                layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None):
        if self.ta_enabled:
            return to_ln
        if not self.ln_first:
            to_ln = self.ln[str(self.idx_state)](to_ln)

        return to_ln

    def pre_ta_ln(self, to_ln: torch.Tensor,
                  layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None,
                  agg: Optional[torch.Tensor] = None):
        if not self.ta_enabled or not self.ln_first:
            return to_ln
        elif self.ta_enabled and self.ln_first:
            if agg is None:
                if self.idx_state == 0 or self.state is None:
                    self.state = to_ln
                agg = self.state
            return self._do_ta_ln(to_ln, agg)
        else:
            return to_ln

    def post_ta_ln(self, to_ln: torch.Tensor,
                   layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None,
                   agg: Optional[torch.Tensor] = None):
        if not self.ta_enabled:
            return to_ln
        if not self.ta_enabled or self.ln_first:
            return to_ln
        elif self.ta_enabled and not self.ln_first:
            if agg is None:
                if self.idx_state == 0 or self.state is None:
                    self.state = to_ln
                agg = self.state
            return self._do_ta_ln(to_ln, agg)
        else:
            return to_ln

    def _do_ta_ln(self, to_ln, agg):
        if self.ta_ln == TransparentAttentionLayerNorm.PostAggregation:
            normed = self.ln[str(self.idx_state)](agg + to_ln)
            return self._set_state_return_normed(normed)
        elif self.ta_ln == TransparentAttentionLayerNorm.PreAggregation:
            normed = agg + self.ln[str(self.idx_state)](to_ln)
            return self._set_state_return_normed(normed)
        raise ValueError(f"{self.ta_ln} was unknown value.")

    def _set_state_return_normed(self, normed):
        self.state = detach_tensor_state(normed)
        return normed

    def __len__(self):
        return len(self.ln)


class MultiLayerNormDelegators(abc.ABC):
    @property
    @abc.abstractmethod
    def delegators(self) -> typing.Dict[int, TorchMultiLayerNormDelegateConfig]:
        pass


class DelegatingTorchMultiLayerNormConfig(Config):
    def __init__(self, delegators: MultiLayerNormDelegators):
        self.delegators = delegators


class DelegatingTorchMultiLayerNorm(LayerNormDelegate):

    def __init__(self, delegating: DelegatingTorchMultiLayerNormConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delegating = torch.nn.ModuleDict({
            i: DelegatingTorchMultiLayerNorm(d) for i, d
            in delegating.delegators.delegators.items()
        })

    def start(self, layer_norm_metadata: MultiLayerLayerNormMetadata):
        self._get_delegate(layer_norm_metadata).start(layer_norm_metadata)

    def _get_delegate(self, layer_norm_metadata) -> TorchMultiLayerNormDelegate:
        d: TorchMultiLayerNormDelegate = typing.cast(TorchMultiLayerNormDelegate,
                                                     self.delegating[str(layer_norm_metadata.layer_idx)])
        return d

    def finish(self, layer_norm_metadata: MultiLayerLayerNormMetadata):
        self._get_delegate(layer_norm_metadata).finish(layer_norm_metadata)

    def pre_ln(self, to_ln: torch.Tensor,
               layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None):
        self._get_delegate(layer_norm_metadata).pre_ln(to_ln, layer_norm_metadata)

    def post_ln(self, to_ln: torch.Tensor,
                layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None):
        self._get_delegate(layer_norm_metadata).post_ln(to_ln, layer_norm_metadata)

    def pre_ta_ln(self, to_ln: torch.Tensor,
                  layer_norm_metadata: Optional[MultiLayerLayerNormMetadata] = None,
                  agg: Optional[torch.Tensor] = None):
        self._get_delegate(layer_norm_metadata).pre_ta_ln(to_ln, layer_norm_metadata)

    def post_ta_ln(self, to_ln: torch.Tensor,
                   layer_norm_metadata: MultiLayerLayerNormMetadata,
                   agg: Optional[torch.Tensor] = None):
        self._get_delegate(layer_norm_metadata).post_ta_ln(to_ln, layer_norm_metadata)
