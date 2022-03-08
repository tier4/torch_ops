import torch
import torch.distributed as dist
import contextlib
#----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.

@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield

def all_reduce(tensor, op):
    if world_size() <= 1:
        return None
    torch.distributed.all_reduce(tensor, op)


def gather_tensors(tensor, async_op=False):
    if world_size() <= 1:
        return tensor
    output = [tensor.clone() for _ in range(world_size())]
    torch.distributed.all_gather(tensor=tensor, tensor_list=output, async_op=async_op)
    return torch.cat(output)



def rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return dist.get_rank()
    return 0


def world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1