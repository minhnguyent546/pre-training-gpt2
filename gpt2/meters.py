import os

import torch
import torch.distributed as dist
from typing import Any

if 'PJRT_DEVICE' in os.environ:
    import torch_xla as xla  # noqa: F401
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

import gpt2.utils as utils


class AverageMeterBase:
    """A base class for average meter."""
    def __init__(
        self,
        name: str = '',
        sum: int | float = 0.0,
        count: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.name = name
        self.sum = sum
        self.count = count
        self.device = device

    @property
    def average(self) -> float:
        try:
            return self.sum / self.count
        except ZeroDivisionError:
            return 0.0

    def update(self, value: int | float, nums: int = 1) -> None:
        self.sum += value * nums
        self.count += nums

    def reduce(self, dst: int) -> None:
        raise NotImplementedError()

    def all_reduce(self) -> None:
        raise NotImplementedError()

    def gather_object(self, dst: int, world_size: int, is_master: bool) -> list[dict[str, Any]] | None:
        raise NotImplementedError()

    def all_gather_object(self, world_size: int) -> list[dict[str, Any]]:
        raise NotImplementedError()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def __repr__(self) -> str:
        str_repr = f'{self.__class__.__name__}('
        if self.name:
            str_repr += f'name={self.name}, '
        str_repr += (
            f'average={self.average}, '
            f'sum={self.sum}, '
            f'count={self.count}'
        )
        if self.device is not None:
            str_repr += f', device={self.device}'
        str_repr += ')'
        return str_repr

    def to_dict(self) -> dict[str, Any]:
        return vars(self)

    def _is_xla_device(self) -> bool:
        return self.device is not None and self.device.type == 'xla'

class AverageMeter(AverageMeterBase):
    """
    A class for working with average meters.

    Support for distributed training with `torch.distributed`.
    """
    def __init__(
        self,
        name: str = '',
        sum: int | float = 0.0,
        count: int = 0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(name, sum, count, device)

    def reduce(self, dst: int) -> None:
        """Perform an in-place reduce."""
        meters_to_reduce = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        # only `Tensor` of process with rank `dst` will be modified in-place,
        # `Tensor` of other processes will remain the same
        dist.reduce(meters_to_reduce, dst=dst, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    def all_reduce(self) -> None:
        """Perform an in-place all reduce."""
        meters_to_reduce = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        dist.all_reduce(meters_to_reduce, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    def gather_object(self, dst: int, world_size: int, is_master: bool) -> list[dict[str, Any]] | None:
        output = [None for _ in range(world_size)] if is_master else None
        object_dict = self.to_dict()
        dist.gather_object(object_dict, output, dst)
        assert output is not None if is_master else output is None
        return output

    def all_gather_object(self, world_size: int) -> list[dict[str, Any]]:
        output = [None for _ in range(world_size)]
        object_dict = self.to_dict()
        dist.all_gather_object(output, object_dict)
        return output

class XLAAverageMeter(AverageMeterBase):
    """
    A class for working with average meters when using XLA devices and running with PJRT runtime.
    """
    def __init__(
        self,
        name: str = '',
        sum: int | float = 0.0,
        count: int = 0,
        device: torch.device | None = None,
    ) -> None:
        if device is not None and not self._is_xla_device():
            raise ValueError(f'Expected device is an XLA device if provided, found {device.type}')
        if device is None:
            device = xm.xla_device()
        super().__init__(name, sum, count, device)

    def all_reduce(self) -> None:
        """Perform an in-place all reduce."""
        meters_to_reduce = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        meters_to_reduce = xm.all_reduce(xm.REDUCE_SUM, meters_to_reduce, scale=1.0 / xr.world_size())
        self.sum, self.count = meters_to_reduce.tolist()

    def all_gather_object(self, world_size: int) -> list[dict[str, Any]]:
        """
        Modified from `torch/distributed/distributed_c10d.py`.

        Note: this function is experimental, use with caution.
        """
        input_tensor, local_size = utils.object_to_tensor(self.to_dict(), self.device)

        object_sizes_tensor = torch.zeros(
            world_size, dtype=torch.long, device=self.device,
        )
        object_size_list = [
            object_sizes_tensor[i].unsqueeze(dim=0) for i in range(world_size)
        ]
        dist.all_gather(object_size_list, local_size)
        max_object_size = int(max(object_size_list).item())  # pyright: ignore

        # resize tensor to max size across all ranks.
        input_tensor.resize_(max_object_size)
        coalesced_output_tensor = torch.empty(
            max_object_size * world_size, dtype=torch.uint8, device=self.device,
        )
        # output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i:max_object_size * (i + 1)]
            for i in range(world_size)
        ]
        dist.all_gather(output_tensors, input_tensor)

        # deserialize outputs back to object.
        object_list = [None for _ in range(world_size)]
        for i, tensor in enumerate(output_tensors):
            tensor = tensor.type(torch.uint8)
            tensor_size = object_size_list[i]
            object_list[i] = utils.tensor_to_object(tensor, tensor_size)
        return object_list
