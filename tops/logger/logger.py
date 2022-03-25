import atexit

import json
import functools
import logging
import torch
import numpy as np
try:
    import wandb
except:
    pass
from abc import ABC, abstractmethod
from argparse import ArgumentError
from pathlib import Path
from typing import List, Union
from torch.utils import tensorboard
from .. import np_make_image_grid, rank
from PIL import Image

_global_step = 0
_epoch = 0


INFO = logging.INFO
WARN = logging.WARN
DEBUG = logging.DEBUG
supported_backends = ["stdout", "json", "tensorboard", "wandb", "image_dumper"]
_output_dir = None

DEFAULT_SCALAR_LEVEL = DEBUG
DEFAULT_LOG_LEVEL = INFO
DEFAULT_LOGGER_LEVEL = INFO

class Backend(ABC):

    @abstractmethod
    def __init__(self):
        pass

    def add_scalar(self, tag, value, **kwargs):
        pass

    def add_dict(self, values, **kwargs):
        pass

    def add_image(self, tag, im: np.ndarray, **kwargs):
        pass

    def log(self, msg, level):
        pass

    def finish(self):
        pass


class TensorBoardBackend(Backend):

    def __init__(self, output_dir: Path):
        output_dir.mkdir(exist_ok=True, parents=True)
        self.writer = tensorboard.SummaryWriter(log_dir=output_dir)
        self.closed = False
    
    def add_scalar(self, tag, value, **kwargs):
        self.writer.add_scalar(tag, value, new_style=True, global_step=_global_step)


    def finish(self):
        if self.closed:
            return
        self.closed = True
        self.writer.flush()
        self.writer.close()
        log("Tensorboard write done.")


class StdOutBackend(Backend):

    def __init__(self, filepath: Path, print_to_file=True) -> None:
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
        self.rootLogger = logging.getLogger()
        self.rootLogger.setLevel(DEFAULT_LOGGER_LEVEL)

        self.consoleHandler = logging.StreamHandler()
        self.consoleHandler.setFormatter(logFormatter)
        self.print_to_file = print_to_file
        if self.print_to_file:
            self.file_handler = logging.FileHandler(filepath)
            self.file_handler.setFormatter(logFormatter)
            self.rootLogger.addHandler(self.file_handler)
        self.rootLogger.addHandler(self.consoleHandler)
        self.closed = False
    
    def add_scalar(self, tag, value, level, **kwargs):
        msg = f"[{_global_step}] {tag}: {value}"
        self.rootLogger.log(level, msg)
    
    def add_dict(self, values, level, **kwargs):
        msg = f"[{_global_step}] "
        for tag, value in values.items():
            msg += f"{tag}: {value:.3f}, "
        self.rootLogger.log(level, msg)
    
    def log(self, msg, level, **kwargs):
        self.rootLogger.log(level, f"[{_global_step}]" + msg)
    
    def finish(self):
        if self.closed:
            return
        self.closed = True
        log("Writing to stdout file.")
        if self.print_to_file:
            self.file_handler.flush()
            self.file_handler.close()
            self.rootLogger.removeHandler(self.file_handler)
        self.rootLogger.removeHandler(self.consoleHandler)
        self.consoleHandler.close()


class ImageDumper(Backend):

    def __init__(self, image_dir: Path) -> None:
        self.image_dir = image_dir

    def add_image(self, tag, im: np.ndarray, **kwargs):
        self.image_dir.joinpath(tag).mkdir(exist_ok=True, parents=True)
        impath = self.image_dir.joinpath(tag, f"{_global_step}.png")
        Image.fromarray(im).save(impath)


class JSONBackend(Backend):

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.file = open(filepath, "a")
        self.closed = False
    
    def add_scalar(self, tag, value, **kwargs):
        self.add_dict({tag: value})
    
    def add_dict(self, values, **kwargs):
        values = {**values, "global_step": _global_step}
        value_str = json.dumps(values) + "\n"
        self.file.write(value_str)
    
    def finish(self):
        if self.closed:
            return
        self.closed = True
        self.file.flush()
        self.file.close()
        log("JSON write done.")

class WandbBackend(Backend):

    def has_previous_run(self):
        return self.output_dir.joinpath("wandb_id.txt").is_file()

    def get_run_id(self):
        run_id_path = self.output_dir.joinpath("wandb_id.txt")
        if run_id_path.is_file():
            with open(run_id_path, "r") as fp:
                uid = str(fp.readlines()[0])
                return uid
        uid = wandb.util.generate_id()
        with open(run_id_path, "w") as fp:
            fp.write(str(uid))
        return uid

    def __init__(self,
            output_dir: Path,
            project,
            experiment_name,
            cfg,
            reinit: bool # Allow multiple wandb instances in the same process
            ) -> None:
        super().__init__()
        self.output_dir = output_dir
        resume = self.has_previous_run()
        self.run = wandb.init(
            project=project,
            dir=str(output_dir),
            name=str(experiment_name),
            config=dict(cfg),
            id=self.get_run_id(),
            resume=resume,
            reinit=reinit)

    def add_scalar(self, tag, value, commit=False, **kwargs):
        wandb.log({tag: value, "global_step":_global_step}, commit=commit)

    def add_dict(self, dictionary: dict, commit=False, **kwargs):
        dictionary["global_step"] = _global_step
        wandb.log(dictionary, commit=commit)

    def add_image(self, tag, im: np.ndarray, **kwargs):
        wandb.log({tag: wandb.Image(im), "global_step": _global_step}, **kwargs)

    def finish(self):
        self.run.finish()
        wandb.finish()
        log("Wandb write done.")

_backends: List[Backend] = [StdOutBackend(None, False)]

def init(
        output_dir, backends,
        experiment_name=None, # Only required for wandb logging
        project=None, # Only required for wandb logging
        cfg=None, # Only required for wandb logging
        reinit=False
        ):
    if rank() != 0:
        return
    global _backends, _output_dir
    for backend in _backends:
        backend.finish()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    _output_dir = output_dir
    _resume()
    _write_metadata()
    _backends = []
    for backend in backends:
        if backend not in supported_backends:
             raise ArgumentError(f"{backend} not in supported. Has to be one of: {', '.join(backends)}")
        if backend == "stdout":
            _backends.append(StdOutBackend(output_dir.joinpath("log.txt")))
        if backend == "tensorboard":
            _backends.append(TensorBoardBackend(output_dir.joinpath("tensorboard")))
        if backend == "json":
            _backends.append(JSONBackend(output_dir.joinpath("scalars.json")))
        if backend == "wandb":
            assert experiment_name is not None
            assert project is not None
            _backends.append(WandbBackend(
                output_dir, project, experiment_name, cfg, reinit))
        if backend == "image_dumper":
            _backends.append(ImageDumper(output_dir.joinpath("images")))


    atexit.register(finish)


def log(msg, level=None):
    if rank() != 0:
        return
    level = DEFAULT_LOG_LEVEL if level is None else level
    for backend in _backends:
        backend.log(msg, level)

@functools.lru_cache(1)
def warn_once(text):
    log(text, level=WARN)


def warn(text):
    log(text, level=WARN)


def _handle_scalar(value):
    if isinstance(value, (float, int)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    if np.issubdtype(value, np.integer):
        return int(value)
    if np.issubdtype(value, np.floating):
        return float(value)
    raise ArgumentError(type(value))

def add_scalar(tag, value, level=None, **kwargs):
    if rank() != 0:
        return
    level = DEFAULT_SCALAR_LEVEL if level is None else level
    for backend in _backends:
        backend.add_scalar(tag, _handle_scalar(value), level=level, **kwargs)


def add_dict(values: dict, level=None, **kwargs):
    level = DEFAULT_LOG_LEVEL if level is None else level
    if rank() != 0:
        return
    for backend in _backends:
        backend.add_dict({tag: _handle_scalar(v) for tag, v in values.items()}, level=level, **kwargs)

def add_images(
        tag, images: Union[np.ndarray, torch.ByteTensor],
        nrow=10, **kwargs):
    """
        images: a single image or list of images. List of images will be saved as a image matrix with nrow rows.
    """
    if rank() != 0:
        return
    assert images.ndim in [3, 4]
    if isinstance(images, np.ndarray):
        assert images.dtype == np.uint8
        if images.ndim == 4:
            images = np_make_image_grid(images, nrow)
        
    if isinstance(images, torch.Tensor):
        assert images.dtype == torch.uint8
        images = images.cpu()
        if images.ndim == 4:
            images = images.permute(0, 2, 3, 1).contiguous().numpy()
            images = np_make_image_grid(images, nrow)
        elif images.ndim == 3:
            images = images.permute(1, 2, 0).contiguous().numpy()
    assert images.ndim == 3
    assert images.dtype == np.uint8
    for backend in _backends:
        backend.add_image(tag, images, **kwargs)

def finish():
    if rank() != 0:
        return
    log("Writing to files.")
    last = None
    _write_metadata()
    for backend in _backends:
        if isinstance(backend, StdOutBackend):
            continue
        backend.finish()
    if last is not None:
        last.finish()


def step(step=1):
    global _global_step
    _global_step += step

def step_epoch():
    global _epoch
    _epoch += 1

def _write_metadata():
    if rank() != 0:
        return
    with open(_output_dir.joinpath("metadata.json"), "w") as fp:
        json.dump(dict(global_step=_global_step, epoch=_epoch), fp)
    log("Metadata write done.")

def _resume():
    global _epoch, _global_step
    metadata_path = _output_dir.joinpath("metadata.json")
    if not metadata_path.is_file():
        return
    with open(metadata_path, "r") as fp:
        data = json.load(fp)
    _epoch = data["epoch"]
    _global_step = data["global_step"]

def epoch():
    return _epoch

def global_step():
    return _global_step


def get_scalars(tag, json_path=None):
    if json_path is None:
        if not any(isinstance(b, JSONBackend) for b  in _backends):
            raise Exception(
                "get_scalars currently only supports a JSONBackend.\
                Note that logger.init() has to be called before retrieving scalars.")
        backend = [b for b in _backends if isinstance(b, JSONBackend)][0]
        json_path = backend.filepath
    with open(json_path, "r") as fp:
        data = json.load(fp)
    data = [x for x in data if tag in x]
    global_step = [x["global_step"] for x in data]
    values = [x[tag] for x in data]
    return global_step, values