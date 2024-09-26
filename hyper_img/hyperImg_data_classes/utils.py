from abc import ABC, abstractmethod

from hyper_img.hyperImg import HyperImg
from hyper_img.hyperImg_segmenter.segmenter import Segmenter

import typing as tp

import os
import numpy as np
import pandas as pd

from collections import defaultdict, Counter


class EmptyHyperImgList(Exception):
    pass


class NeedHyperImgSubclass(Exception):
    pass


class NeedSequenceOrPath(Exception):
    pass


class ThisMethodDoesNotExist(Exception):
    pass


class DifferentTargetVariableName(Exception):
    pass


class DifferentCameraSetting(Exception):
    pass


class FeatureDataLenError(Exception):
    pass


class SameFactorsError(Exception):
    pass


def all_tiff_cube_img(path: str) -> list[str]:
    img_names: list[str] = list()
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name = os.path.join(filename)
            if name.split('.')[-1] != 'tiff':
                continue
            if name.split('.')[0].split('_')[-1] != 'cube':
                continue
            img_names.append(dirname + '/' + name)
    return img_names

def sort_hyper(hyper_imgs):
    tg_variable_dct = defaultdict(list)
    for img in hyper_imgs:
        tg_variable_dct[img.get_factors()].append(img)
    for k in tg_variable_dct:
        tg_variable_dct[k] = sorted(tg_variable_dct[k], key=lambda x: x.object_name)
    hyper_lst = []
    for k in tg_variable_dct:
        hyper_lst.extend(tg_variable_dct[k])
    return hyper_lst

def get_list_hyper_img(class_name: type,
                    segmenter: Segmenter,
                    path: str | None = None,
                    seq_names: tp.Sequence | None = None,
                    *args, **kwargs) -> list[HyperImg]:
    """
    Create the list of hyperspectral images.
    Args:
        class_name (type): images class.
        path (str | None): images path. Defaults to None.
        seq_names (pd.DataFrame | None): sequence with paths. Defaults to None.
        filter (tp.Callable[[str], bool] | None, optional): filter function. Defaults to None.
        same_samples (tp.Sequence): same samples target variable names.
        args: args for HyperImg object.
        kwargs: kwargs for HyperImg object.
    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the result is an empty list. Most likely an error in the way.
        NeedSequenceOrPath: path is None and seq_names is None.
    Returns:
        list[HyperImg]: the list of hyperspectral images.
    """ 
    if not issubclass(class_name, HyperImg):
        raise NeedHyperImgSubclass
    
    if path is None and seq_names is None:
        raise NeedSequenceOrPath
    
    lst: list['HyperImg'] = list()
    if path is not None:
        for name in all_tiff_cube_img(path):
            hyper_img: HyperImg = class_name(name, segmenter=segmenter, *args, **kwargs)
            lst.append(hyper_img)

    if seq_names is not None:
        for name in seq_names:
            hyper_img = class_name(name, segmenter=segmenter, *args, **kwargs)
            lst.append(hyper_img)

    if len(lst) == 0:
        raise EmptyHyperImgList

    return sort_hyper(lst)
