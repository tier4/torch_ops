from pathlib import Path
from typing import Any, List

import torch
from easydict import EasyDict

from tops.logger import global_step, log
from tops.logger.logger import _write_metadata
from tops.utils.torch_utils import get_device, rank

_checkpoint_dir: Path | None = None
_models: (
    dict[
        str,
        torch.nn.parallel.DistributedDataParallel
        | torch.nn.Module
        | torch.amp.GradScaler
        | torch.optim.Optimizer
        | dict
        | EasyDict,
    ]
    | None
) = None


def init(checkpoint_dir: Path) -> None:
    global _checkpoint_dir
    _checkpoint_dir = checkpoint_dir


def load_checkpoint(
    checkpoint_path: Path | None = None,
    load_best: bool = False,
    map_location: torch.device | None = None,
) -> dict:
    """
    checkpoint_path has to be a directory path, filepath. If none, tops has to be initialized (tops.init).
    """
    if map_location is None:
        map_location = get_device()
    if checkpoint_path is None:
        checkpoint_path = _checkpoint_dir
        if _checkpoint_dir is None:
            raise ValueError(
                "Both the provided checkpoint_path and global checkpoint_dir is None."
                + "You have to initialize tops or provide a checkpoint."
            )
    assert checkpoint_path is not None
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        log(f"Loaded checkpoint from {checkpoint_path}")
        return ckpt
    checkpoint_dir = checkpoint_path
    if load_best:
        checkpoint_path = checkpoint_dir.joinpath("best_model.ckpt")
    else:
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"No checkpoint folder exists in: {checkpoint_dir.absolute()}"
            )

        checkpoints = get_ckpt_paths(checkpoint_dir)
        if len(checkpoints) == 0:
            raise FileNotFoundError(f"No checkpoints in folder: {checkpoint_path}")
        checkpoint_path = checkpoints[-1]
    assert checkpoint_path.is_file(), "This should not be reachable."
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Did not find file: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    log(f"Loaded checkpoint from {checkpoint_path}")
    return ckpt


def get_ckpt_paths(checkpoint_dir: Path) -> List[Path]:
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoints = [x for x in checkpoint_dir.glob("*.ckpt") if x.stem != "best_model"]
    checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return checkpoints


def save_checkpoint(
    state_dict: dict,
    checkpoint_dir: Path | None = None,
    is_best: bool = False,
    max_keep: int = 1,
) -> None:
    if rank() != 0:
        return
    if checkpoint_dir is None:
        assert _checkpoint_dir is not None
        checkpoint_dir = _checkpoint_dir
    checkpoint_dir.parent.mkdir(exist_ok=True, parents=True)
    previous_checkpoint_paths = get_ckpt_paths(checkpoint_dir)
    if is_best:
        torch.save(state_dict, checkpoint_dir.joinpath("best_model.ckpt"))
        log(f"Saved model to: {checkpoint_dir.joinpath('best_model.ckpt')}")
    checkpoint_path = checkpoint_dir.joinpath(f"{global_step()}.ckpt")
    if checkpoint_path.is_file():
        log(f"Checkpoint already exists: {checkpoint_path}.")
        return
    torch.save(state_dict, checkpoint_path)
    log(f"Saved model to: {checkpoint_path}")
    if len(previous_checkpoint_paths) > max_keep:
        previous_checkpoint_paths[0].unlink()


def has_checkpoint(checkpoint_dir: Path | None = None) -> bool:
    if checkpoint_dir is None:
        assert _checkpoint_dir is not None
        checkpoint_dir = _checkpoint_dir
    checkpoint_dir = Path(checkpoint_dir)
    num_checkpoints = len(list(checkpoint_dir.glob("*.ckpt")))
    return num_checkpoints > 0


def register_models(models: dict) -> None:
    global _models
    for _, model in models.items():
        if isinstance(model, (dict, EasyDict)):
            continue
        if not hasattr(model, "state_dict"):
            raise ValueError("The model has to have a state_dict")
        if not hasattr(model, "load_state_dict"):
            raise ValueError("The model has to have load_state_dict")
    _models = models


def save_registered_models(other_state: dict | None = None, **kwargs: Any) -> None:
    assert _models is not None
    state_dict = {}
    for key, model in _models.items():
        if isinstance(model, (dict, EasyDict)):
            state_dict[key] = model
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict[key] = model.module.state_dict()
        else:
            state_dict[key] = model.state_dict()
    if other_state:
        assert all(key not in state_dict for key in other_state)
        state_dict.update(other_state)
    save_checkpoint(state_dict, **kwargs)
    _write_metadata()


def load_registered_models(
    checkpoint_path: Path | None = None,
    map_location: torch.device | None = None,
    strict: bool = True,
) -> dict:
    assert _models is not None
    state_dict = load_checkpoint(
        checkpoint_path=checkpoint_path, map_location=map_location
    )
    for key, state in state_dict.items():
        if key in _models:
            v = _models[key]
            if isinstance(v, (dict, EasyDict)):
                v.update(state)
            else:
                if isinstance(v, torch.nn.parallel.DistributedDataParallel):
                    v.module.load_state_dict(state, strict=strict)
                elif isinstance(v, torch.nn.Module):
                    v.load_state_dict(state, strict=strict)
                elif isinstance(v, (torch.optim.Optimizer, torch.amp.GradScaler)):
                    v.load_state_dict(state)
                else:
                    raise ValueError(f"Unexpected model type: {type(v)}")
    return {k: v for k, v in state_dict.items() if key not in _models}
