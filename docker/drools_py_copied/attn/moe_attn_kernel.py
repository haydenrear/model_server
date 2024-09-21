from typing import Tuple, Union, Optional

import torch

from codegen.generated_config_models import ScoreClassifierEmbeddingSizeDenseFfn, ScoreClassifierOutDimDense, \
    NumLayersDenseFfnScoreClassifier, NumFfnLayersDenseFfnScoreClassifier, RouterEmbeddingSizeDenseFfn, \
    NumLayersDenseFfnRouterScoreClassifier, NumLayersDenseRouterScoreClassifier, OutputProbabilitiesTypeRouter, \
    RouterOutDimDense, ScoreClassifierInDimDense, ScoreClassifierKernelTypes, \
    NumExpertsMoeScoreClassifierKernel, InDimAttnKernelConfig, MoeRouterPoolingType, InDimLinearLayer, OutDimLinearLayer
from drools_py.attn.attn_kernel_config import AttnKernelConfig
from drools_py.attn.base_fourier_attn_kernel import BaseFourierAttnKernel
from drools_py.configs.config import Config, ConfigType
from drools_py.configs.config_factory import ConfigFactory
from drools_py.configs.config_models import NumExpertsMoe
from drools_py.ffn.dense_ffn import DenseFfn
from drools_py.ffn.ffn_config import DenseFfnConfig, LinearLayerConfig, create_dense_layers
from drools_py.output_probabilities.output_probabilities import OutputProbabilitiesConfigFactory
from drools_py.pooling.pooling_configs import PoolerConfig, PoolerFactory
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn
from python_util.logger.logger import LoggerFacade
from python_util.torch_utils.complex_torch import complex_to_2d, to_complex_from_2d


class ScoreKernelConfig(Config):
    def __init__(self, dense_ffn: list[DenseFfnConfig]):
        self.dense_ffn = dense_ffn

    @classmethod
    @autowire_fn()
    def score_kernel_config(cls,
                            config_type: ConfigType,
                            embed_dim: ScoreClassifierEmbeddingSizeDenseFfn,
                            out_dim_dense: ScoreClassifierOutDimDense,
                            num_layers_dense: NumLayersDenseFfnScoreClassifier,
                            num_layers_ffn_dense: NumFfnLayersDenseFfnScoreClassifier,
                            in_dim: ScoreClassifierInDimDense):
        dense_layers = create_dense_layers(config_type,
                                           in_dim.config_option * 2,
                                           num_layers_ffn_dense.config_option,
                                           out_dim_dense.config_option * 2,
                                           num_layers_dense.config_option)
        return ScoreKernelConfig(dense_layers)


class ScoreKernel(torch.nn.Module):

    def __init__(self, config: ScoreKernelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.kernel = torch.nn.ModuleList([DenseFfn(d) for d in self.config.dense_ffn])

    def forward(self, to_compute_kernel: torch.Tensor) -> torch.Tensor:
        o = to_compute_kernel
        for d in self.kernel:
            o = d(o)

        return o


class ScoreKernelFactory(ConfigFactory):

    def __init__(self, kernel_config: ScoreKernelConfig):
        super().__init__(kernel_config)
        self.kernel_config = kernel_config
        self.model = None

    def create(self, **kwargs):
        return ScoreKernel(self.kernel_config)


class RouterConfig(Config):
    def __init__(self, n_experts: NumExpertsMoe,
                 proba: OutputProbabilitiesConfigFactory,
                 dense_ffn_config: list[DenseFfnConfig],
                 embedding_layer: LinearLayerConfig,
                 probability_pool: PoolerFactory):
        self.probability_pool = probability_pool
        self.embedding_layer = embedding_layer
        self.dense_ffn_config = dense_ffn_config
        self.proba = proba
        self.n_experts = n_experts

    @classmethod
    @autowire_fn()
    def score_classifier_router_config(cls,
                                       config_type: ConfigType,
                                       router_out_dim_dense: RouterOutDimDense,
                                       num_experts: NumExpertsMoeScoreClassifierKernel,
                                       output_proba: OutputProbabilitiesTypeRouter,
                                       router_dense_layers: NumLayersDenseRouterScoreClassifier,
                                       router_dense_ffn_layers: NumLayersDenseFfnRouterScoreClassifier,
                                       in_dim: ScoreClassifierInDimDense,
                                       moe_router_pool_type: MoeRouterPoolingType):
        return RouterConfig(
            num_experts,
            OutputProbabilitiesConfigFactory.output_probabilities_config_factory(output_proba),
            create_dense_layers(config_type, in_dim.config_option * 2, router_dense_ffn_layers.config_option,
                                router_out_dim_dense.config_option * 2 * num_experts.config_option,
                                router_dense_layers.config_option),
            LinearLayerConfig.linear_layer_config(InDimLinearLayer(in_dim.config_option * 2 * num_experts.config_option),
                                                  OutDimLinearLayer(in_dim.config_option * 2 * num_experts.config_option)),
            PoolerFactory(PoolerConfig(moe_router_pool_type))
        )


class Router(torch.nn.Module):
    def __init__(self, router_config: RouterConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_proba = router_config.proba.create(dim=-1)
        self.dense = torch.nn.ModuleList([DenseFfn(d) for d in router_config.dense_ffn_config])
        self.linear = router_config.embedding_layer.to_linear_layer()
        self.pool = router_config.probability_pool.create()
        self.config = router_config

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        :param input_values: The input fourier transform s x 2 x f, where 2 is i and r.
        :return: probability for i and r together to multiply by.
        """
        o = input_values
        for d in self.dense:
            o = d(o)

        linear_result = self.linear(o)
        proba_result = self.router_proba(linear_result)
        proba_shape = [i for i in proba_result.shape[:-1]]
        proba_shape.extend([self.config.n_experts.config_option, input_values.shape[-1]])
        resulting = proba_result.reshape(proba_shape)
        resulting_t = resulting.permute(3, 0, 1, 2, 4) # out is [n_experts, batch_size, n_attn_heads, seq_len, seq_len * 2
        return resulting_t


class MixtureOfExpertsScoreKernelConfig(AttnKernelConfig):

    def __init__(self,
                 score_kernel_factory: ScoreKernelFactory,
                 router: RouterConfig,
                 n_kernels: NumExpertsMoe,
                 in_dim: ScoreClassifierInDimDense,
                 attn_kernel: ScoreClassifierKernelTypes,
                 probability_pool: PoolerFactory):
        super().__init__(attn_kernel, InDimAttnKernelConfig(in_dim.config_option * 2))
        self.probability_pool = probability_pool
        self.n_kernels = n_kernels
        self.score_kernel_factory = score_kernel_factory
        self.router = router

    @classmethod
    @autowire_fn()
    def moe_score_kernel_config(cls,
                                config_type: ConfigType,
                                score_kernel_factory: ScoreKernelFactory,
                                router: RouterConfig,
                                n_kernels: NumExpertsMoeScoreClassifierKernel,
                                in_dim: ScoreClassifierInDimDense,
                                attn_kernel: ScoreClassifierKernelTypes,
                                moe_router_pool_type: MoeRouterPoolingType):
        return MixtureOfExpertsScoreKernelConfig(score_kernel_factory, router, n_kernels, in_dim, attn_kernel,
                                                 PoolerFactory(PoolerConfig(moe_router_pool_type)))


class MixtureOfExpertsScoreKernel(BaseFourierAttnKernel):
    def __init__(self, moe_score_kernel: MixtureOfExpertsScoreKernelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        LoggerFacade.info("Creating MOE score kernel.")
        self.num_kernels = moe_score_kernel.n_kernels.config_option
        LoggerFacade.info("Creating router for MOE.")
        self.moe_router = Router(moe_score_kernel.router)
        LoggerFacade.info("Creating score kernel factories.")
        self.score_kernels = torch.nn.ModuleDict({
            str(i): moe_score_kernel.score_kernel_factory.create()
            for i in range(moe_score_kernel.n_kernels.config_option)
        })

    def forward(self,
                input_value: torch.Tensor,
                return_kernels: bool = False,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        This will take the input data, a complex value and cast it to float, expanding in 2d. Then it will input
        that into the router to determine how much of each kernel to apply. This will output 2d for real and imaginary
        as well. So then for each kernel function, input the casted float and output the kernel, which will also
        be 2d for real and imaginary. Apply the kernel functions according to the router, and then, before returning,
        cast back to complex value form the 2d r and i.

        # TODO: perhaps the power spectral density would be better suited, however in this case there is less info
            and who says the dense FFN can't learn the rotation anyway. Another way to do would be ifft and to real
            with geometry and then same back.
        :param tgt_mask:
        :param input_value: The fourier transform of the data
        :param return_kernels:
        :return:
        """
        # TODO: This is similar to attention where the MOE router getting the scores to multiply by.
        input_value = complex_to_2d(input_value)
        router_probs = self.moe_router(input_value)
        to_multiply = []

        for i in range(len(self.score_kernels)):
            to_multiply.append(self.score_kernels[str(i)](input_value))

        stacked = torch.stack(to_multiply)
        for i, t in enumerate(stacked):
            router_prob = router_probs[i]
            # get probability multiplier per item for each score kernel
            # t -> [1, 2, 128, 256] router_prob -> [1, 2, 128, 256]
            next_value = torch.matmul(t.transpose(-1, -2), router_prob)
            # multiply by probability -> next_value -> [1, 2, 256, 256] * input_value [1, 2, 128, 256]
            # TODO: can try a pointwise connection here as well, with complex LN
            input_value = torch.matmul(input_value, next_value)

        next_out = to_complex_from_2d(input_value)

        reshaped_router_probs = self._router_prob(router_probs)

        if return_kernels:
            return next_out, reshaped_router_probs, stacked
        else:
            return next_out, reshaped_router_probs, None

    def _router_prob(self, router_probs):
        to_reshape = [i for i in router_probs.shape[:-1]]
        to_reshape.extend([2, router_probs.shape[-1] // 2])
        reshaped_router_probs = router_probs.reshape(to_reshape)
        reshaped_router_probs = torch.nn.functional.normalize(reshaped_router_probs, dim=-2)
        reshaped_router_probs = torch.mean(reshaped_router_probs, dim=0)
        reshaped_router_probs = torch.mean(reshaped_router_probs, dim=-2)
        reshaped_router_probs = torch.nn.functional.normalize(reshaped_router_probs)
        return reshaped_router_probs

    def _ln_complex(self, before, router_probs, size_value):
        to_reshape = [i for i in router_probs.shape[:-1]]
        to_reshape.extend([2, size_value])
        reshaped = router_probs.reshape(to_reshape)
        real_value = self.real_norm(router_probs[:, :, 0:size_value])
        complex_value = self.complex_norm(router_probs[:, :, size_value:])
        reshaped[:, :, 0] = real_value
        reshaped[:, :, 1] = complex_value
        router_probs = reshaped.reshape(before)
        return router_probs

