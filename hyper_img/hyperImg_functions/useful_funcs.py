from hyper_img.hyperImg import HyperImg

import numpy as np
import pandas as pd

from collections import Counter

import typing as tp


def change_target_variable_names(hyper_imges: tp.Sequence[HyperImg],
                                 names: dict[str, str],
                                 inplace: bool = True) -> list[HyperImg] | None:
    """
    Change target variable names for all samples in hyper_imges.
    Args:
        hyper_imdes (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        names (dict[str, str]): name resolution dictionary (from keys to values).
        inplace (bool): inplace operation or not. Defaults False.
    Returns:
        list[HyperImg] | None
    """
    if inplace:
        for i, img in enumerate(hyper_imges):
            if img.target_variable in names.keys():
                hyper_imges[i].target_variable = names[img.target_variable]
    else:
        new_hyper_imges = []
        for img in hyper_imges:
            new_hyper_imges.append(img)
            if img.target_variable in names.keys():
                new_hyper_imges[-1].target_variable = names[img.target_variable]
        return new_hyper_imges


def get_count_group(hyper_imges: tp.Sequence[HyperImg]) -> dict[str, int]:
    """
    Get the number of samples of each group.
    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
    Returns:
        dict[str, int]: dictionary with number of elements for each groups.
    """
    return dict(Counter([img.target_variable for img in hyper_imges]))


def rename(hyper_imges: tp.Sequence[HyperImg],
           object_names: tp.Sequence,
           inplace: bool = True) -> list[HyperImg] | None:
    """
    Change object names sequentially.
    Args:
        hyper_imges(tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        object_names: object names.
    Returns:
        list[HyperImg] | None
    """
    if inplace:
        for i, name in enumerate(object_names):
            hyper_imges[i].object_name = name
    else:
        new_hyper_imges = []
        for i, img in enumerate(hyper_imges):
            new_hyper_imges.append(img)
            new_hyper_imges[-1].object_name = object_names[i]
        return new_hyper_imges
