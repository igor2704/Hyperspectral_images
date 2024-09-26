from hyper_img.hyperImg import HyperImg
from hyper_img.hyperImg_data_classes.data import HyperData
from hyper_img.hyperImg_segmenter.segmenter import Segmenter

import typing as tp
import numpy as np
import pandas as pd

from collections import defaultdict


def _groupby_id(hyper_imges: tp.Sequence[HyperImg]) -> dict[str, list[HyperImg]]:
    hyper_dct = defaultdict(list)
    for img in hyper_imges:
        hyper_dct[img.object_name].append(img)
    return hyper_dct


def _groupby_id_mean(hyper_imges: tp.Sequence[HyperImg]) -> tp.Sequence[HyperImg]:
    hyper_dct = _groupby_id(hyper_imges)
    mean_hyper_imges = list()
    for img_id in hyper_dct:
        hyper_sum = hyper_dct[img_id][0]
        for i, img in enumerate(hyper_dct[img_id]):
            if i == 0:
                continue
            hyper_sum = hyper_sum + img
        hyper_sum = hyper_sum / len(hyper_dct[img_id])
        mean_hyper_imges.append(hyper_sum)
    return mean_hyper_imges


class HyperDataMedian(HyperData):
    def __init__(self,
                 class_name: type,
                 segmenter: Segmenter,
                 mean_aggregate: bool = False,
                 path: str | None = None,
                 seq_names: tp.Sequence | None = None,
                 *args, **kwargs) -> None:

        super().__init__(class_name, segmenter, path, seq_names,
                         *args, **kwargs)
        self.feature_name = 'Median'
        self.mean_aggregate = mean_aggregate
        if self.mean_aggregate:
            self.hyper_imgs_lst = _groupby_id_mean(self.hyper_imgs_lst)

    def _feature_extract(self, sample:HyperImg) -> list[np.ndarray]:
        return [sample.get_medians()]
