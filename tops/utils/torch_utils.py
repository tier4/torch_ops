import random
from typing import Any
import numpy as np
import torch
import tqdm
import torch.distributed as dist
import torch
from contextlib import contextmanager
from time import time
from easydict import EasyDict

AMP_enabled = False
_seed = 0

def rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def set_AMP(value: bool):
    global AMP_enabled
    AMP_enabled = value


def AMP():
    return AMP_enabled


def _to_cuda(element):
    return element.to(get_device(), non_blocking=True)


def to_cuda(elements):
    if isinstance(elements, tuple) or isinstance(elements, list):
        return [_to_cuda(x) for x in elements]
    if isinstance(elements, dict):
        return {k: _to_cuda(v) for k,v in elements.items()}
    return _to_cuda(elements)


def get_device() -> torch.device:
    return torch.device(f"cuda:{rank()}") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed: int):
    global _seed
    _seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_seed(seed:  int):
    return _seed



def zero_grad(model):
    """
    Reduce overhead of optimizer.zero_grad (read+write).
    """
    for param in model.parameters():
        param.grad = None

def set_requires_grad(module: torch.nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def tqdm(iterator, *args, **kwargs):
    if rank() == 0:
        return tqdm.tqdm(iterator, *args, **kwargs)
    return iterator


def trange(iterator, *args, **kwargs):
    if rank() == 0:
        return tqdm.trange(iterator, *args, **kwargs)
    return iterator


@contextmanager
def timeit(desc):
    try:
        torch.cuda.synchronize()
        t0 = time()
        yield
    finally:
        torch.cuda.synchronize()
        print(f"({desc}) total time: {time() - t0:.1f}")



def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    # Adapted from: https://github.com/NVlabs/stylegan3
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    if isinstance(inputs, dict):
        outputs = module(**inputs)
    else:
        assert isinstance(inputs, (tuple, list))
        outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs




def cuda_stream_wrap(stream):
    if torch.cuda.is_available():
        return torch.cuda.stream(stream)

    @contextmanager
    def placeholder(stream):
        try:
            yield
        finally:
            pass
    return placeholder(stream)


class DataPrefetcher:
    """
        A dataloader wrapper to prefetch batches to GPU memory.
    """

    def __init__(self,
                 loader: torch.utils.data.DataLoader,
                 image_gpu_transforms: torch.nn.Module):
        self.original_loader = loader
        self.stream = None
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream(device=get_device())
        self.loader = iter(self.original_loader)
        self.image_gpu_transforms = image_gpu_transforms
        self.it = 0

    @torch.no_grad()
    def _preload(self):
        try:
            self.batch = next(self.loader)
            self.stop_iteration = False
        except StopIteration:
            self.stop_iteration = True
            return
        with cuda_stream_wrap(self.stream):
            if isinstance(self.batch, dict):
                for key, item in self.batch.items():
                    self.batch[key] = to_cuda(item).float()
            if isinstance(self.batch, (tuple)):
                self.batch = tuple(to_cuda(x)  for x in self.batch)
            self.batch = self.image_gpu_transforms(self.batch)

    def __len__(self):
        return len(self.original_loader)

    def __next__(self):
        return self.next()

    def next(self):
        if torch.cuda.is_available():
            torch.cuda.current_stream(device=get_device()).wait_stream(self.stream)
        if self.stop_iteration:
            raise StopIteration
        container = self.batch
        self._preload()
        return container

    def __iter__(self):
        self.it += 1
        self.loader = iter(self.original_loader)
        self._preload()
        return self
    
    def __getattr__(self, name: str) -> Any:
        return getattr(self.original_loader, name)



----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.
# From https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/torch_utils/misc.py

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, window_size=0.5, **kwargs):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = get_seed()
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1