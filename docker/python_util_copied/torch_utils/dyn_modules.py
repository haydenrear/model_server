import abc
from typing import TypeVar, Generic

import torch.nn

from python_util.torch_utils.pytorch_util import insert_at_indices, drop_indices

T = TypeVar('T')


class TensorModificationArgs:
    def __init__(self,
                 indices: torch.Tensor,
                 reinitialization: torch.Tensor):
        self.reinitialization = reinitialization
        self.indices = indices


class TensorModification(abc.ABC):

    def __init__(self,
                 tensor_modification_args: TensorModificationArgs):
        self.tensor_modification_args = tensor_modification_args

    @abc.abstractmethod
    def modify_tensor(self, in_tensor: torch.Tensor) -> torch.Tensor:
        pass


class AddIndexTensorModification(TensorModification):

    def __init__(self,
                 tensor_modification_args: TensorModificationArgs):
        super().__init__(tensor_modification_args)

    def modify_tensor(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return insert_at_indices(in_tensor, self.tensor_modification_args.indices, 0)


class RemoveIndexTensorModification(TensorModification):

    def __init__(self,
                 tensor_modification_args: TensorModificationArgs):
        super().__init__(tensor_modification_args)

    def modify_tensor(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return drop_indices(in_tensor, self.tensor_modification_args.indices)


class ReInitializeAtIndices(TensorModification):

    def __init__(self,
                 tensor_modification_args: TensorModificationArgs):
        super().__init__(tensor_modification_args)

    def modify_tensor(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return insert_at_indices(in_tensor,
                                 self.tensor_modification_args.indices,
                                 self.tensor_modification_args.reinitialization)


class ModuleModifier(Generic[T], abc.ABC):
    @abc.abstractmethod
    def modify_module(self, in_module: T) -> T:
        pass


class LinearModuleModifier(ModuleModifier[torch.nn.Linear]):
    def modify_module(self, in_module: torch.nn.Linear) -> torch.nn.Linear:
        pass
