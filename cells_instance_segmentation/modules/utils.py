"""Module with utils for model and dataset"""

from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from cv2 import cv2


def rle_decode(mask_rle, shape=(520, 704)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


def rle_encode(img: np.ndarray) -> str:
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_masks(
    fn: Union[str, Path],
    predictor: Callable,
    thresholds: Tuple[float, float, float] = (0.15, 0.35, 0.55),
    min_pixels: Tuple[int, int, int] = (75, 150, 75),
) -> List[str]:
    im = cv2.imread(str(fn))
    pred = predictor(im)
    pred_class = torch.mode(pred['instances'].pred_classes)[0]
    take = pred['instances'].scores >= thresholds[int(pred_class)]
    pred_masks = pred['instances'].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()
    res = []
    used = np.zeros(im.shape[:2], dtype=int)
    for mask in pred_masks:
        mask = mask * (1 - used)
        if mask.sum() >= min_pixels[int(pred_class)]:
            used += mask
            res.append(rle_encode(mask))
    return res
