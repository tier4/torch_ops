import numpy as np


def np_make_image_grid(images, nrow, pad=2, row_major=True, pad_value=0):
    height, width = images[0].shape[:2]
    ncol = int(np.ceil(len(images) / nrow))
    for idx in range(len(images)):
        assert images[idx].shape == images[0].shape, (images[idx].shape, images[0].shape, idx)
    if isinstance(pad, int):
        pad = (pad, pad)

    if not row_major:
        t = nrow
        nrow = ncol
        ncol = t
    im_result = np.zeros(
        (nrow * (height + pad[0]), ncol * (width + pad[1]), images[0].shape[-1]), dtype=images[0].dtype
    ) + pad_value
    im_idx = 0
    for row in range(nrow):
        for col in range(ncol):
            if im_idx == len(images):
                break
            im = images[im_idx]
            if not row_major:
                im = images[row + col*nrow]
            im_idx += 1
            rstart = row * (pad[0] + height)
            rend = row * (pad[0] + height) + height
            cstart = col * (pad[1] + width)
            cend = col * (pad[1] + width) + width
            im_result[rstart:rend, cstart:cend, :] = im
    return im_result
