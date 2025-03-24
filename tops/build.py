from pathlib import Path

from tops.checkpointer.checkpointer import init as init_checkpointer
from tops.logger.logger import init as _init_logger


def init(
    output_dir: Path,
    backends: list[str] | None = None,
    checkpoint_dir: Path | None = None,
) -> None:
    """
    Initialize the tops framework with logging and checkpointing capabilities.

    This function sets up the logging system and checkpointer with appropriate
    directories. It creates the output directory if it doesn't exist.

    Args:
        output_dir: Path to the directory where outputs (logs, checkpoints) will be saved.
        backends: List of logging backends to use. Defaults to ["stdout", "json", "tensorboard"]
            if None is provided.
        checkpoint_dir: Path to the directory where checkpoints will be saved.
            If None, defaults to a 'checkpoints' subdirectory within output_dir.

    Returns:
        None
    """  # noqa: E501
    if backends is None:
        backends = ["stdout", "json", "tensorboard"]
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    _init_logger(
        output_dir / "logs",
        backends,
    )
    if checkpoint_dir is None:
        checkpoint_dir = output_dir / "checkpoints"
    init_checkpointer(checkpoint_dir)
