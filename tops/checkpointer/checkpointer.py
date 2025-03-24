from pathlib import Path
from typing import TypeAlias

import torch
from easydict import EasyDict

from tops.logger import global_step
from tops.logger.logger import _write_metadata
from tops.utils.torch_utils import get_device, rank

RegisteredModule: TypeAlias = (
    torch.nn.parallel.DistributedDataParallel
    | torch.nn.Module
    | torch.amp.GradScaler
    | torch.optim.Optimizer
    | dict
    | EasyDict
)
RegisteredModules: TypeAlias = dict[str, RegisteredModule]
_checkpoint_dir: Path | None = None
_registered_modules: RegisteredModules | None = None


def init(checkpoint_dir: Path) -> None:
    """
    Initialize the checkpointer with a directory to save checkpoints.

    This function sets the global checkpoint directory that will be used by default
    when saving or loading checkpoints if no specific path is provided.

    Args:
        checkpoint_dir: Path to the directory where checkpoints will be saved.

    Returns:
        None
    """  # noqa: E501
    global _checkpoint_dir
    _checkpoint_dir = checkpoint_dir


def load_checkpoint(
    checkpoint_path: Path | None = None,
    map_location: torch.device | None = None,
    weights_only: bool = False,
) -> dict:
    """
    Load a checkpoint from disk.

    This function loads a checkpoint from the specified path or from the global checkpoint directory.
    If a directory is provided instead of a file, it will load the latest checkpoint from that directory.

    Args:
        checkpoint_path: Path to the checkpoint file or directory containing checkpoints.
                         If None, uses the global checkpoint directory.
        map_location: Device to map the checkpoint to when loading.
                      If None, uses the current device.
        weights_only: If True, only loads the model weights without optimizer states
                      and other metadata.

    Returns:
        A dictionary containing the loaded checkpoint state.

    Raises:
        ValueError: If both the provided checkpoint_path and global checkpoint_dir are None.
        FileNotFoundError: If the checkpoint file or directory does not exist,
                          or if no checkpoint files are found in the directory.
    """  # noqa: E501
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
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=weights_only
        )
        return checkpoint
    checkpoint_dir = checkpoint_path
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"No checkpoint directory exists at: {checkpoint_dir.absolute()}"
        )

    checkpoints = get_checkpoint_paths(checkpoint_dir)
    if len(checkpoints) == 0:
        raise FileNotFoundError(f"No checkpoint files at {checkpoint_dir}")
    checkpoint_path = checkpoints[-1]
    return torch.load(
        checkpoint_path, map_location=map_location, weights_only=weights_only
    )


def get_checkpoint_paths(checkpoint_dir: Path | None = None) -> list[Path]:
    """
    Get a sorted list of checkpoint paths from the specified directory.

    This function retrieves all checkpoint files (with .ckpt extension) from the given directory
    and sorts them based on the numerical value in the filename. The sorting assumes checkpoint
    filenames end with a number (e.g., 'checkpoint_1000.ckpt').

    Args:
        checkpoint_dir: Path to the directory containing checkpoints.
                        If None, uses the global checkpoint directory.

    Returns:
        A list of Path objects pointing to checkpoint files, sorted by checkpoint number.

    Raises:
        ValueError: If both the provided checkpoint_dir and global checkpoint_dir are None,
                   or if the provided path is not a directory.
        FileNotFoundError: If the checkpoint directory does not exist.
    """  # noqa: E501
    if checkpoint_dir is None:
        if _checkpoint_dir is None:
            raise ValueError(
                "Both the provided checkpoint_dir and global checkpoint_dir is None."
                + "You have to initialize tops or provide a checkpoint directory."
            )
        checkpoint_dir = _checkpoint_dir
    if not checkpoint_dir.is_dir():
        raise ValueError(f"{checkpoint_dir} is not a directory or does not exist")
    checkpoint_paths = [path for path in checkpoint_dir.glob("*.ckpt")]
    checkpoint_paths.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return checkpoint_paths


def save_checkpoint(
    state_dict: dict,
    checkpoint_dir: Path | None = None,
    max_keep: int = -1,
) -> None:
    """
    Save a model checkpoint to disk.

    This function saves the provided state dictionary as a checkpoint file in the specified directory.
    The checkpoint filename is based on the current global step. Only the process with rank 0
    will save the checkpoint to avoid duplicate writes in distributed training.

    Args:
        state_dict: Dictionary containing the model state to save.
        checkpoint_dir: Directory where the checkpoint will be saved.
                       If None, uses the global checkpoint directory.
        max_keep: Maximum number of checkpoints to keep. If > 0, the oldest checkpoint
                 will be deleted when this limit is exceeded. If -1 (default), keeps all checkpoints.

    Raises:
        ValueError: If both the provided checkpoint_dir and global checkpoint_dir are None.

    Note:
        - Only the process with rank 0 will save checkpoints.
        - The checkpoint filename format is "{global_step}.ckpt".
        - If the checkpoint directory doesn't exist, it will be created.
    """  # noqa: E501
    if rank() != 0:
        return
    if checkpoint_dir is None:
        if _checkpoint_dir is None:
            raise ValueError(
                "Both the provided checkpoint_dir and global checkpoint_dir is None."
                + "You have to initialize tops or provide a checkpoint directory."
            )
        checkpoint_dir = _checkpoint_dir
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    prev_checkpoint_paths = get_checkpoint_paths(checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"{global_step()}.ckpt"
    torch.save(state_dict, checkpoint_path)
    if max_keep > 0 and (len(prev_checkpoint_paths) > max_keep):
        prev_checkpoint_paths[0].unlink()


def has_checkpoint(checkpoint_dir: Path | None = None) -> bool:
    """
    Check if there are any checkpoints in the specified directory.

    This function verifies if any checkpoint files exist in the given directory.
    If no directory is provided, it uses the global checkpoint directory.

    Args:
        checkpoint_dir: Directory to check for checkpoints.
                       If None, uses the global checkpoint directory.

    Returns:
        bool: True if at least one checkpoint exists, False otherwise.

    Raises:
        ValueError: If both the provided checkpoint_dir and global checkpoint_dir are None,
                   or if the checkpoint directory does not exist or is not a directory.

    Note:
        - Checkpoint files are expected to have the ".ckpt" extension.
        - The function uses get_checkpoint_paths internally to find checkpoint files.
    """  # noqa: E501
    if checkpoint_dir is None:
        if _checkpoint_dir is None:
            raise ValueError(
                "Both the provided checkpoint_dir and global checkpoint_dir is None."
                + "You have to initialize tops or provide a checkpoint directory."
            )
        checkpoint_dir = _checkpoint_dir
    if not checkpoint_dir.is_dir():
        raise ValueError(f"{checkpoint_dir} is not a directory or does not exist")
    return len(get_checkpoint_paths(checkpoint_dir=checkpoint_dir)) > 0


def register_modules(models: RegisteredModules, skip_unsupported: bool = False) -> None:
    """
    Register models for checkpoint saving and loading.

    This function registers a dictionary of models that can be saved or loaded together.
    The registered models are stored in a global variable and can be accessed by
    save_registered_models and load_registered_models functions.

    Args:
        models: Dictionary mapping names to model objects. Supported types include:
               - torch.nn.Module
               - torch.nn.parallel.DistributedDataParallel
               - torch.optim.Optimizer
               - torch.amp.GradScaler
               - dict
               - EasyDict
        skip_unsupported: If True, silently skip any objects with unsupported types.
                         If False (default), raise ValueError for unsupported types.

    Raises:
        ValueError: If any object in the models dictionary has an unsupported type
                   and skip_unsupported is False.

    Note:
        This function must be called before using save_registered_models or
        load_registered_models functions.
    """
    global _registered_modules
    for _, model in models.items():
        if not isinstance(
            model,
            (
                dict,
                EasyDict,
                torch.nn.parallel.DistributedDataParallel,
                torch.nn.Module,
                torch.optim.Optimizer,
                torch.amp.GradScaler,
            ),
        ):
            if skip_unsupported:
                continue
            raise ValueError(f"Cannot register {type(model)} objects (not supported)")
    _registered_modules = models


def save_registered_modules(
    checkpoint_dir: Path | None = None,
    max_keep: int = -1,
) -> None:
    """
    Save all registered modules to a checkpoint file.

    This function saves the state dictionaries of all registered modules to a checkpoint file.
    For dictionaries and EasyDict objects, the entire object is saved.
    For DistributedDataParallel modules, the wrapped module's state_dict is saved.
    For other supported objects (nn.Module, Optimizer, GradScaler), their state_dict is saved.

    Args:
        checkpoint_dir: Directory where the checkpoint will be saved.
                       If None, uses the default directory.
        max_keep: Maximum number of checkpoints to keep in the directory.
                 If -1 (default), keeps all checkpoints.

    Raises:
        ValueError: If no modules have been registered using register_modules.

    Note:
        This function also writes metadata about the current training state.
    """  # noqa: E501
    if _registered_modules is None:
        raise ValueError("You have to register models first using register_models")
    state_dict = {}
    for key, model in _registered_modules.items():
        if isinstance(model, (dict, EasyDict)):
            state_dict[key] = model
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict[key] = model.module.state_dict()
        else:
            state_dict[key] = model.state_dict()
    save_checkpoint(state_dict, checkpoint_dir=checkpoint_dir, max_keep=max_keep)
    _write_metadata()


def load_registered_modules(
    checkpoint_path: Path | None = None,
    map_location: torch.device | None = None,
    strict: bool = True,
) -> None:
    """
    Load all registered modules from a checkpoint file.

    This function loads the state dictionaries from a checkpoint file into the registered modules.
    For dictionaries and EasyDict objects, the checkpoint data is merged into the existing object.
    For DistributedDataParallel modules, the state is loaded into the wrapped module.
    For other supported objects (nn.Module, Optimizer, GradScaler), their state_dict is loaded.

    Args:
        checkpoint_path: Path to the checkpoint file to load.
                        If None, uses the latest checkpoint in the default directory.
        map_location: Device where the tensors should be loaded.
                     If None, tensors are loaded to their original device.
        strict: Whether to strictly enforce that the keys in the module's state_dict
               match the keys in the checkpoint. Only applies to nn.Module objects.

    Raises:
        ValueError: If no modules have been registered using register_modules.
        FileNotFoundError: If no checkpoint file is found at the specified path.
    """  # noqa: E501
    if _registered_modules is None:
        raise ValueError("You have to register models first using register_models")
    state_dict = load_checkpoint(
        checkpoint_path=checkpoint_path, map_location=map_location
    )
    for key, state in state_dict.items():
        if key in _registered_modules:
            v = _registered_modules[key]
            if isinstance(v, (dict, EasyDict)):
                v.update(state)
            else:
                if isinstance(v, torch.nn.parallel.DistributedDataParallel):
                    v.module.load_state_dict(state, strict=strict)
                elif isinstance(v, torch.nn.Module):
                    v.load_state_dict(state, strict=strict)
                else:
                    v.load_state_dict(state)
