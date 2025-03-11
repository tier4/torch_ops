import torch
import torch.distributed as dist


def rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return dist.get_rank()
    return 0


def world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1
