import abc
import dataclasses
import enum
import logging
import uuid
from typing import TypeVar, Generic

import torch

from python_di.configs.bean import bean
# from python_di.configs.bean import bean
from python_di.configs.component import component
from python_di.env.base_env_properties import DEFAULT_PROFILE
from python_di.inject.profile_composite_injector.composite_injector import ProfileScope
from python_util.logger.logger import LoggerFacade
from drools_py.serialize.serializable_types import SerializableEnum

T = TypeVar("T")


class EnumConfigOption(SerializableEnum, metaclass=enum.EnumMeta):
    @classmethod
    def default(cls, **kwargs):
        logging.error(f"Must override default method in EnumConfigOption for {cls}")


class ConfigOption(Generic[T], abc.ABC):

    def __init__(self, config_option: T = None):
        if isinstance(config_option, ConfigOption):
            self._config_option = config_option.config_option
        else:
            self._config_option = config_option

    @property
    def config_option(self) -> T:
        return self._config_option

    @config_option.setter
    def config_option(self, config_option: T):
        self._config_option = config_option

    def __hash__(self):
        return hash(self.config_option)

    def _perform_op(self, other, op_cb):
        if hasattr(other, 'config_option'):
            return op_cb(self.config_option, other.config_option)
        else:
            return op_cb(self.config_option, other)

    def __eq__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val == other_val)

    def __radd__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val + other_val)

    def __rmul__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val * other_val)

    def __rsub__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val - other_val)

    def __gt__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val > other_val)

    def __ge__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val >= other_val)

    def __le__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val <= other_val)

    def __lt__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val < other_val)

    def __add__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val + other_val)

    def __sub__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val - other_val)

    def __mul__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val * other_val)

    def __divmod__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val.__divmod__(other_val))

    def __mod__(self, other):
        return self._perform_op(other, lambda self_val, other_val: self_val % other_val)

    def __int__(self):
        return self._try_convert(int)

    def __bool__(self):
        return self._try_convert(bool)

    def __float__(self):
        return self._try_convert(float)

    def __str__(self):
        return self._try_convert(str)

    def __abs__(self, *args, **kwargs):  # real signature unknown
        return self.config_option.__abs__(*args, **kwargs)

    def __ceil__(self, *args, **kwargs):  # real signature unknown
        return self.config_option.__ceil__(*args, **kwargs)

    def __floordiv__(self, *args, **kwargs):  # real signature unknown
        """ Return self//value. """
        return self.config_option.__floordiv__(*args, **kwargs)

    def __ifloordiv__(self, other):  # real signature unknown
        """ Return self//value. """
        return self._perform_op(other, lambda self_val, other_val: self_val.__ifloordiv__(other_val))

    def __rfloordiv__(self, other):  # real signature unknown
        """ Return self//value. """
        return self._perform_op(other, lambda self_val, other_val: self_val.__rfloordiv__(other_val))

    def __neg__(self, *args, **kwargs):  # real signature unknown
        """ -self """
        return self.config_option.__neg__(*args, **kwargs)

    def __floor__(self, *args, **kwargs):  # real signature unknown
        """ Return the floor as an Integral. """
        return self.config_option.__floor__(*args, **kwargs)

    def _try_convert(self, convert_to):
        try:
            return convert_to(self.config_option)
        except Exception as e:
            LoggerFacade.error(f"Attempted to convert {type(self)} to {convert_to} but received exception {e}.")


class InDim(ConfigOption[int]):

    def __init__(self, config_option=10):
        super().__init__(config_option)


class EdgeDim(ConfigOption[int]):

    def __init__(self, config_option=10):
        super().__init__(config_option)


class EdgeAttr(ConfigOption[int]):

    def __init__(self, config_option=10):
        super().__init__(config_option)


class NTimesteps(ConfigOption[int]):

    def __init__(self, config_option=12):
        super().__init__(config_option)


class GradientClippingValue(ConfigOption[float]):

    def __init__(self, config_option=0.5):
        super().__init__(config_option)


class NumLayers(ConfigOption[int]):

    def __init__(self, config_option=2):
        super().__init__(config_option)


class K(ConfigOption[int]):

    def __init__(self, config_option=8):
        super().__init__(config_option)


class NumAttnHeads(ConfigOption[int]):

    def __init__(self, config_option=2):
        super().__init__(config_option)


class NumFrequencyBands(ConfigOption[int]):
    def __init__(self, config_option=32):
        super().__init__(config_option)


class BaseDim(ConfigOption[int]):

    def __init__(self, config_option=10):
        super().__init__(config_option)

    @classmethod
    @bean(profile="test", self_factory=True, scope=ProfileScope)
    def build_test_config(cls, **kwargs):
        return cls(10)

    @classmethod
    def build_test_prop(cls, **kwargs):
        return 10

    @classmethod
    @bean(profile="validation", self_factory=True, scope=ProfileScope)
    def build_validation_config(cls, **kwargs):
        return cls(1280)

    @classmethod
    def build_validation_prop(cls, **kwargs):
        return 1280


class OutDim(BaseDim):

    def __init__(self, config_option=10):
        super().__init__(config_option)


class TensorShape(ConfigOption[list[int]]):

    def __init__(self, config_option=None):
        super().__init__(config_option)


class NSteps(ConfigOption[int]):

    def __init__(self, config_option=1):
        super().__init__(config_option)


class EmbeddingSize(BaseDim):
    def __init__(self, config_option=10):
        super().__init__(config_option)


class Divisor(ConfigOption[int]):
    def __init__(self, config_option=2):
        super().__init__(config_option)


class SequenceLength(ConfigOption[int]):
    def __init__(self, config_option=512):
        super().__init__(config_option)


class IncludeBias(ConfigOption[bool]):

    def __init__(self, config_option=True):
        super().__init__(config_option)


class DoEnable(ConfigOption[bool]):

    def __init__(self, config_option=True):
        super().__init__(config_option)


class DoSkipConnections(ConfigOption[bool]):

    def __init__(self, config_option=True):
        super().__init__(config_option)


class TransparentAttentionEnabled(ConfigOption[bool]):

    def __init__(self, config_option=True):
        super().__init__(config_option)


Dim = int


class Frozen(ConfigOption[bool]):

    def __init__(self, config_option=True):
        super().__init__(config_option)


class ShuffleData(ConfigOption[bool]):

    def __init__(self, config_option=False):
        super().__init__(config_option)


class BeamSearchIsEncoderDecoder(ConfigOption[bool]):

    def __init__(self, config_option=True):
        super().__init__(config_option)


class StepSize(ConfigOption[int]):

    def __init__(self, config_option=10):
        super().__init__(config_option)


class VocabSize(ConfigOption[int]):

    def __init__(self, config_option=33556):
        super().__init__(config_option)


class EosToken(ConfigOption[int]):

    def __init__(self, config_option=-1):
        super().__init__(config_option)


class PadToken(ConfigOption[int]):

    def __init__(self, config_option=-1):
        super().__init__(config_option)


class StartToken(ConfigOption[int]):

    def __init__(self, config_option=-1):
        super().__init__(config_option)


class BatchSize(ConfigOption[int]):

    def __init__(self, config_option=1):
        super().__init__(config_option)


class TopK(ConfigOption[int]):

    def __init__(self, config_option=1):
        super().__init__(config_option)


class BeamWidth(ConfigOption[int]):

    def __init__(self, config_option=10):
        super().__init__(config_option)


class TopP(ConfigOption[float]):

    def __init__(self, config_option=0.9):
        super().__init__(config_option)


class AccumulateGradientsBatchSize(BatchSize):

    def __init__(self, config_option=50):
        super().__init__(config_option)


class DatasetUri(ConfigOption[str]):

    def __init__(self, config_option="graelo/wikipedia"):
        super().__init__(config_option)


class SaveDatasetUri(DatasetUri):

    def __init__(self, config_option="/Users/hayde/IdeaProjects/drools/train_data/test_dataset"):
        super().__init__(config_option)


class Device(ConfigOption[str]):

    def __init__(self, config_option="cpu"):
        super().__init__(config_option)


class DType(ConfigOption[torch.dtype]):

    def __init__(self, config_option: torch.dtype = torch.float32):
        super().__init__(config_option)


class LayerId(ConfigOption[str]):

    def __init__(self, config_option=str(uuid.uuid4())):
        if config_option == '':
            config_option = str(uuid.uuid4())
        super().__init__(config_option)


class LearningRate(ConfigOption[float]):

    def __init__(self, config_option=3e-5):
        super().__init__(config_option)


class MaxLearningRate(LearningRate):

    def __init__(self, config_option=3e-5):
        super().__init__(config_option)


class MinLearningRate(LearningRate):

    def __init__(self, config_option=1e-6):
        super().__init__(config_option)


class LearningRateModeTypes(EnumConfigOption):
    min = enum.auto()


class LearningRateMode(ConfigOption[LearningRateModeTypes]):
    def __init__(self, config_option=LearningRateModeTypes.min):
        super().__init__(config_option)


class NumIterationsLr(ConfigOption[int]):

    def __init__(self, config_option=2000):
        super().__init__(config_option)


class WarmupIterationsLr(NumIterationsLr):

    def __init__(self, config_option=2000):
        super().__init__(config_option)


class NumIterationsFirstRestartLr(NumIterationsLr):

    def __init__(self, config_option=2000):
        super().__init__(config_option)


class EpochMultiplierRestart(ConfigOption[int]):

    def __init__(self, config_option=3):
        super().__init__(config_option)


class WeightDecay(ConfigOption[float]):

    def __init__(self, config_option=3e-5):
        super().__init__(config_option)


class LearningRateGamma(ConfigOption[float]):

    def __init__(self, config_option=0.1):
        super().__init__(config_option)


class ThresholdValue(ConfigOption[float]):

    def __init__(self, config_option=0.8):
        super().__init__(config_option)


class IncludeSelfLoops(ConfigOption[bool]):

    def __init__(self, config_option=True):
        super().__init__(config_option)


class LayerNormFirst(ConfigOption[bool]):

    def __init__(self, config_option=True):
        super().__init__(config_option)


class TransparentAttentionLayerNorm(EnumConfigOption):
    PreAggregation = enum.auto()
    PostAggregation = enum.auto()


class TransparentAttentionLayerNormConfigOption(ConfigOption[TransparentAttentionLayerNorm]):
    def __init__(self,
                 config_option: TransparentAttentionLayerNorm = TransparentAttentionLayerNorm.PostAggregation):
        super().__init__(config_option)


class NumEpochs(ConfigOption[int]):

    def __init__(self, config_option=3):
        super().__init__(config_option)


class MaxEpochs(ConfigOption[int]):

    def __init__(self, config_option=3):
        super().__init__(config_option)


class StepsPerEpochLr(ConfigOption[int]):

    def __init__(self, config_option=1000):
        super().__init__(config_option)


class ConcurrentTasks(ConfigOption[int]):

    def __init__(self, config_option=3):
        super().__init__(config_option)


class BatchFirst(ConfigOption[bool]):

    def __init__(self, config_option=False):
        super().__init__(config_option)


class StreamDataset(ConfigOption[bool]):

    def __init__(self, config_option=False):
        super().__init__(config_option)


class BaseGenericBool(ConfigOption[bool]):

    def __init__(self, config_option=False):
        super().__init__(config_option)


class NumGroups(ConfigOption[int]):

    def __init__(self, config_option=2):
        super().__init__(config_option)


class NumSamples(ConfigOption[int]):

    def __init__(self, config_option=5):
        super().__init__(config_option)


class LayerNormEps(ConfigOption[float]):

    def __init__(self, config_option=1e-5):
        super().__init__(config_option)


class Dropout(ConfigOption[float]):

    def __init__(self, config_option=0.0):
        super().__init__(config_option)


class ModelId(ConfigOption[str]):

    def __init__(self, config_option=str(uuid.uuid4())):
        super().__init__(config_option)


class CheckpointPath(ConfigOption[str]):

    def __init__(self, config_option=""):
        super().__init__(config_option)


class LoggerPath(ConfigOption[str]):

    def __init__(self, config_option=str(uuid.uuid4())):
        super().__init__(config_option)


class TrainingLoggerPath(LoggerPath):

    def __init__(self, config_option=str(uuid.uuid4())):
        super().__init__(config_option)


class TensorboardLoggerPath(TrainingLoggerPath):

    def __init__(self, config_option=str(uuid.uuid4())):
        super().__init__(config_option)


class IterationsPerCheckpoint(ConfigOption[int]):

    def __init__(self, config_option: int = 32):
        super().__init__(config_option)


class ModelClassName(ConfigOption[str]):

    def __init__(self, config_option=str(uuid.uuid4())):
        super().__init__(config_option)


class Kwargs(ConfigOption[dict]):
    def __init__(self, config_option: dict = None):
        super().__init__(config_option)


class LayerNormKwargs(ConfigOption[dict]):
    def __init__(self, config_option: dict = None):
        super().__init__(config_option)


class CreateKwargs(LayerNormKwargs):

    def __init__(self, config_option: dict = None):
        super().__init__(config_option)


class ForwardKwargs(LayerNormKwargs):

    def __init__(self, config_option: dict = None):
        super().__init__(config_option)


class DatasetLoadKwargs(LayerNormKwargs):

    def __init__(self, config_option: dict = None):
        super().__init__(config_option)


class PythonModulePath(ConfigOption[str]):
    def __init__(self, config_option: str = ""):
        super().__init__(config_option)


class ModuleImport(PythonModulePath):
    def __init__(self, config_option: str = ""):
        super().__init__(config_option)


class TokenizerId(ConfigOption[str]):

    def __init__(self, config_option=str(uuid.uuid4())):
        super().__init__(config_option)


class Temperature(ConfigOption[float]):

    def __init__(self, config_option=0.2):
        super().__init__(config_option)


class GraphParserConvolutionKernel(ConfigOption[int]):

    def __init__(self, config_option=3):
        super().__init__(config_option)


class NumMixturesBayes(ConfigOption[int]):

    def __init__(self, config_option=1):
        super().__init__(config_option)


class NumExpertsMoe(ConfigOption[int]):

    def __init__(self, config_option=1):
        super().__init__(config_option)


class ModuleEnabled(ConfigOption[bool]):

    def __init__(self, config_option=True):
        super().__init__(config_option)


class LayerNormEnabled(ModuleEnabled):

    def __init__(self, config_option=True):
        super().__init__(config_option)


class FinalLayerNorm(LayerNormEnabled):

    def __init__(self, config_option=False):
        super().__init__(config_option)
