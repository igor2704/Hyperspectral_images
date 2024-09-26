from hyper_img.hyperImg import HyperImg
from hyper_img.hyperImg_data_classes.data import HyperData
from hyper_img.hyperImg_segmenter.segmenter import Segmenter

import typing as tp
import numpy as np
import pandas as pd

from collections import defaultdict


def select_pixels(hyper_img: HyperImg,
                  left_percent: float,
                  right_percent: float) -> np.ndarray:
    all_pixels = hyper_img.get_all_values().T
    all_pixels_sum = np.sum(all_pixels, axis=1)

    left_p = left_percent
    right_p = 1 - right_percent

    left_q = np.quantile(all_pixels_sum, left_p)
    right_q = np.quantile(all_pixels_sum, right_p)

    filtered_pixels = [pixel for pixel in all_pixels
                       if left_q < pixel.sum() < right_q]

    return np.array(filtered_pixels)


def get_buckets(pixels: np.ndarray,
                number_buckets: int | None) -> np.ndarray:
    if number_buckets is None:
        n_buckets = len(pixels)
    else:
        n_buckets = min(number_buckets, len(pixels))
    buckets_lst = [[] for _ in range(n_buckets)]
    for i, pix in enumerate(pixels):
        buckets_lst[i % n_buckets].append(pix)
    buckets = [[] for _ in range(n_buckets)]
    for i, b in enumerate(buckets_lst):
        buckets[i] = np.mean(buckets_lst[i], axis=0)
    buckets = np.array(buckets)
    return buckets


class HyperDataPixels(HyperData):
    def __init__(self,
                 class_name: type,
                 segmenter: Segmenter,
                 number_buckets: int | None = None,
                 left_percent: float = 0.05,
                 right_percent: float = 0.05,
                 path: str | None = None,
                 seq_names: tp.Sequence | None = None,
                 *args, **kwargs) -> None:
        self.number_buckets = number_buckets
        self.left_percent = left_percent
        self.right_percent = right_percent
        super().__init__(class_name, segmenter, path, seq_names,
                          *args, **kwargs)
        self.feature_name = f'Bucket ({number_buckets})'

    def _feature_extract(self, sample:HyperImg) -> list[np.ndarray]:
        return list(get_buckets(select_pixels(sample, self.left_percent, self.right_percent),
                                              self.number_buckets))
