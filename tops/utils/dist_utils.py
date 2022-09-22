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


def all_gather_uneven(x):
    if world_size() <= 1:
        return x
    device = torch.device(f"cuda:{rank()}")

    local_size = torch.tensor(x.size(), device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)

    max_size = max(all_sizes)
    size_diff = max_size.item() - local_size.item()
    if size_diff:
        padding = torch.zeros(size_diff, device=device, dtype=x.dtype)
        x = torch.cat((x, padding))

    all_xs_padded = [torch.zeros_like(x) for _ in range(ws)]
    dist.all_gather(all_xs_padded, x)

    all_xs = []
    for x, size in zip(all_xs_padded, all_sizes):
        all_xs.append(x[:size])
    return torch.cat(all_xs)


def rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return dist.get_rank()
    return 0


def world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1