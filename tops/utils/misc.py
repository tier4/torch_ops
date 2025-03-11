import os
import sys
from contextlib import contextmanager

import numpy as np


def np_make_image_grid(
    images: list[np.ndarray],
    nrow: int,
    pad: int = 2,
    row_major: bool = True,
    pad_value: int = 0,
) -> np.ndarray:
    height, width = images[0].shape[:2]
    ncol = int(np.ceil(len(images) / nrow))
    for idx in range(len(images)):
        assert images[idx].shape == images[0].shape, (
            images[idx].shape,
            images[0].shape,
            idx,
        )
    if isinstance(pad, int):
        pads = (pad, pad)

    if not row_major:
        t = nrow
        nrow = ncol
        ncol = t
    im_result = (
        np.zeros(
            (nrow * (height + pads[0]), ncol * (width + pads[1]), images[0].shape[-1]),
            dtype=images[0].dtype,
        )
        + pad_value
    )
    im_idx = 0
    for row in range(nrow):
        for col in range(ncol):
            if im_idx == len(images):
                break
            im = images[im_idx]
            if not row_major:
                im = images[row + col * nrow]
            im_idx += 1
            rstart = row * (pads[0] + height)
            rend = row * (pads[0] + height) + height
            cstart = col * (pads[1] + width)
            cend = col * (pads[1] + width) + width
            im_result[rstart:rend, cstart:cend, :] = im
    return im_result


@contextmanager
def capture_stdout():
    try:
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        sys.stdout = old_stdout
