from pathlib import Path

from tops.checkpointer.checkpointer import init as init_checkpointer
from tops.logger.logger import init as _init_logger


def init(
    output_dir: Path | str,
    backends: list[str] | None = None,
    checkpoint_dir: Path | str | None = None,
) -> None:
    """
    Initialize the tops framework with logging and checkpointing capabilities.

    This function sets up the logging system and checkpointer with appropriate
    directories. It creates the output directory if it doesn't exist.

    Args:
        output_dir: Path to the directory where outputs (logs, checkpoints) will be saved.
            Will be created if it doesn't exist.
        backends: List of logging backends to use. Defaults to ["stdout", "json", "tensorboard"]
            if None is provided. Valid backends include "stdout", "json", and "tensorboard".
        checkpoint_dir: Path to the directory where checkpoints will be saved.
            If None, defaults to a 'checkpoints' subdirectory within output_dir.
            Will be converted to a Path object if provided as a string.

    Returns:
        None

    Example:
        >>> from tops.build import init
        >>> init("./outputs", backends=["stdout", "tensorboard"])
    """  # noqa: E501
    if backends is None:
        backends = ["stdout", "json", "tensorboard"]
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    _init_logger(
        output_dir / "logs",
        backends,
    )
    if checkpoint_dir is None:
        checkpoint_dir = output_dir / "checkpoints"
    else:
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)
    init_checkpointer(checkpoint_dir)
