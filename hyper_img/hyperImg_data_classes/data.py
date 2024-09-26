from abc import ABC, abstractmethod

from hyper_img.hyperImg import HyperImg
from hyper_img.hyperImg_segmenter.segmenter import Segmenter

import typing as tp

import os
import numpy as np
import pandas as pd

from collections import defaultdict, Counter

from hyper_img.hyperImg_data_classes.utils import EmptyHyperImgList, NeedHyperImgSubclass, NeedSequenceOrPath, \
                  DifferentTargetVariableName, DifferentCameraSetting, FeatureDataLenError, SameFactorsError
from hyper_img.hyperImg_data_classes.utils import get_list_hyper_img
from hyper_img.hyperImg_data_classes.useful_funcs import change_target_variable_names, get_count_group,\
                         rename, write_json_with_all_values


class HyperData(ABC):

    def __init__(self,
                class_name: type,
                segmenter: Segmenter,
                path: str | None = None,
                seq_names: tp.Sequence | None = None,
                *args, **kwargs) -> None:
        """
        Args:
            class_name (type): images class.
            path (str | None): images path. Defaults to None.
            seq_names (pd.DataFrame | None): sequence with paths. Defaults to None.
            filter (tp.Callable[[str], bool] | None, optional): filter function. Defaults to None.
            same_samples (tp.Sequence): same samples target variable names.
            args: args for HyperImg object.
            kwargs: kwargs for HyperImg object.
        """
        self.class_name = class_name
        self.segmenter = segmenter
        self.path = path
        self.seq_names = seq_names
        self.args = args
        self.kwargs = kwargs

        self.feature_name = 'feature'

        self.hyper_imgs_lst = get_list_hyper_img(self.class_name,
                                                 self.segmenter,
                                                 self.path,
                                                 self.seq_names,
                                                 *self.args,
                                                 **self.kwargs)

        if len(self.hyper_imgs_lst) == 0:
            raise EmptyHyperImgList

        if not issubclass(type(self.hyper_imgs_lst[0]), HyperImg):
            raise NeedHyperImgSubclass

        self.camera_sensitive = self.hyper_imgs_lst[0].camera_sensitive
        self.camera_begin_wavelenght = self.hyper_imgs_lst[0].camera_begin_wavelenght
        self.factors = list(self.hyper_imgs_lst[0].factors.keys())

        for img in self.hyper_imgs_lst:
            if img.camera_sensitive != self.camera_sensitive or img.camera_begin_wavelenght != self.camera_begin_wavelenght:
                raise DifferentCameraSetting
            if list(img.factors.keys()) != self.factors:
                raise SameFactorsError

        self.channels_len = len(self.hyper_imgs_lst[0].get_medians())

        self.hyper_table = self._get_table()

    @abstractmethod
    def _feature_extract(self, sample:HyperImg) -> list[np.ndarray]:
        pass

    def _get_table(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: table with samples with feature for each channel.
        """
        features = list()
        for sample in self.hyper_imgs_lst:
            f = self._feature_extract(sample)
            if len(f[0]) != self.channels_len:
                raise FeatureDataLenError
            f_lst = []
            for el in f:
                f_lst.append(list(el) + [sample.object_name]
                             + list(sample.factors.values()) + [sample.get_factors()])
            features.extend(f_lst)
        wl_columns = ['wl_' + str(wl) 
                      for wl in np.arange(0, self.channels_len) * self.camera_sensitive + self.camera_begin_wavelenght]
        return pd.DataFrame(features,
                            columns= wl_columns
                                    + ['Object name'] + self.factors + ['all_factors'])

    def create_table_annotation_df(self,
                                   name: str = 'Image Name',
                                   object_name: str = 'PlantNumber',
                                   black_calibr_name: str = 'Black calibration data',
                                   white_calibr_name: str = 'White calibration data') -> pd.DataFrame:
        """
        Create table for annotation.
        Args:
            name (str): column title with image name.
            black_calibr_name (str): column title with black image for calibration name.
            white_calibr_name (str): column title with white image for calibration name.

        Raises:
            NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
            EmptyHyperImgList: the sequence of images was empty.

        Returns:
            pd.DataFrame: created table
        """

        table_dct: dict[str, list[tp.Any]] = defaultdict(list)

        for h_img in self.hyper_imgs_lst:
            table_dct[name].append(h_img.path.split('/')[-1])
            if h_img.object_name is not None:
                table_dct[object_name].append(h_img.object_name)
            table_dct[black_calibr_name].append(h_img.black_calibration_img_name)
            table_dct[white_calibr_name].append(h_img.white_calibration_img_name)

        return pd.DataFrame(table_dct)

    def append(self, sample: HyperImg) -> None:
        pass

    def insert(self, index: int, sample: HyperImg) -> None:
        pass

    def remove(self, sample: HyperImg) -> None:
        pass

    def extend(self, data: 'HyperData') -> None:
        pass

    def __len__(self):
        return len(self.hyper_imgs_lst)

    def get_count_groups(self) -> dict[str, int]:
        return get_count_group(self.hyper_imgs_lst)

    def write_json_with_all_segment_values(self,
                                           json_path: str) -> None:
        return write_json_with_all_values(self.hyper_imgs_lst,
                                          json_path)
