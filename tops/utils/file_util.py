import pathlib
import validators
import torch
import os
import errno
import sys
import warnings
from urllib.parse import urlparse
from pathlib import Path
from hashlib import md5


def download_file(url, progress=True, check_hash=False, file_name=None, subdir=None):
    r"""Downloads and caches file to TORCH CACHE
        Adapted from: torch.hub.load_state_dict_from_url

    Args:
        url (string): URL of the object to download
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (string, optional): name for the downloaded file. Filename from `url` will be used if not set.
    Example:
        >>> state_dict = tops.download_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    hub_dir = torch.hub.get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    if subdir is not None:
        filename = os.path.join(subdir, filename)
    cached_file = Path(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = torch.hub.HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        cached_file.parent.mkdir(exist_ok=True, parents=True)
        torch.hub.download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file


def is_image(impath: Path):
    return impath.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp", ".webp"]


def is_video(impath: Path):
    return impath.suffix.lower() in [".mp4", ".webm", ".avi"]


def load_file_or_url(path: str, map_location=None, md5sum: str = None):
    filepath = pathlib.Path(path)
    if not torch.cuda.is_available():
        map_location = torch.device("cpu")
    if filepath.is_file():
        return torch.load(path, map_location=map_location)
    validators.url(path)
    filepath = download_file(path)
    if md5sum is not None:
        with open(filepath, "rb") as fp:
            cur_md5sum = md5(fp.read()).hexdigest()
        assert md5sum == cur_md5sum, \
            f"The downloaded file does not match the given md5sum. Downloaded file md5: {cur_md5sum}, expected: {md5sum}"
    return torch.load(filepath, map_location=map_location)
