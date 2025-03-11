from pathlib import Path

from tops.checkpointer.checkpointer import init as init_checkpointer
from tops.logger.logger import init as _init_logger


def init(
    output_dir: Path,
    logging_backend: list[str] = ["stdout", "json", "tensorboard", "image_dumper"],
    checkpoint_dir: Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    _init_logger(
        output_dir.joinpath("logs"),
        logging_backend,
    )
    if checkpoint_dir is None:
        checkpoint_dir = output_dir.joinpath("checkpoints")
    init_checkpointer(checkpoint_dir)
