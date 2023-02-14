from hyper_img.hyperImg import HyperImg

import typing as tp

import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class EmptyHyperImgList(Exception):
    pass


class NeedHyperImgSubclass(Exception):
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


def get_list_hyper_img(path: str, class_name: type,
                       filter: tp.Callable[[str], bool] | None = None,
                       threshold_value: float = 25,
                       savgol_par: tuple[int, int] = (9, 3),
                       target_varible_name: str = 'Target Varible') -> list[HyperImg]:
    """
    Create the list of hyperspectral images.
    Args:
        path (str): images path.
        class_name (type): images class.
        filter (tp.Callable[[str], bool] | None, optional): filter function. Defaults to None.
        threshold_value (float, optional): threshold. Defaults to 25.
        savgol_par (tuple[int, int], optional): parametrs for savgol method. Defaults to (9, 3).
        target_varible_name (str, optional): name of target varible. Defaults to 'Target Varible'.

    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the result is an empty list. Most likely an error in the way.

    Returns:
        list[HyperImg]: the list of hyperspectral images.
    """
    
    if not issubclass(class_name, HyperImg):
        raise NeedHyperImgSubclass
    
    lst: list['HyperImg'] = list()
    for name in all_tiff_cube_img(path):
        hyper_img: HyperImg = class_name(name, threshold_value, savgol_par, 
                                         target_varible_name)
        if filter is None or filter(hyper_img.target_varible):
            lst.append(hyper_img)
    
    if len(lst) == 0:
        raise EmptyHyperImgList
    
    return lst


def get_df_graphics_medians_wavelenght(hyper_imges: tp.Sequence[HyperImg],
                                       special_sample_path: str = '') -> pd.DataFrame:
    """
    Create DataFrame for graphics medians versus wavelength.
    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        special_sample_path (str, optional): subtracts from each median vector the median vector of the image along this path.. Defaults to ''.

    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.
    Returns:
        pd.DataFrame: DataFrame for graphics medians versus wavelength.
    """

    if len(hyper_imges) == 0:
        raise EmptyHyperImgList
    
    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass

    x_axis: tp.List[int] = list(np.arange(0, 138) * 4 + 450)
    points: tp.List[tp.Any] = list()
    if special_sample_path:
        special_sample = type(hyper_imges[0])(special_sample_path, 
                                              hyper_imges[0]._threshold_value,
                                              hyper_imges[0].savgol_par,
                                              hyper_imges[0].target_varible_name)

    for sample_number, sample in enumerate(hyper_imges):

        if sample.path == special_sample_path and special_sample_path:
            continue

        if not special_sample_path:
            point = zip(x_axis, sample.medians)
        else:
            point = zip(x_axis, sample.medians - special_sample.medians)

        for p in point:
            points.append([p[0], p[1], sample_number, sample.target_varible])

    return pd.DataFrame(points, columns=['Wavelength', 'Median',
                                         'Sample', hyper_imges[0].target_varible_name])


def get_df_medians(hyper_imges: tp.Sequence[HyperImg], 
                   filter_value: str = '') -> pd.DataFrame:
    """
    Create DataFrame of medians.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        filter_value (str, optional): if target varible = filter_value, that this sample is skipped. Defaults to ''.

    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.
    Returns:
        pd.DataFrame: DataFrame of medians.
    """

    if len(hyper_imges) == 0:
        raise EmptyHyperImgList
    
    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass

    return pd.DataFrame([list(sample.medians) + [sample.target_varible] for sample in hyper_imges
                         if sample.target_varible != filter_value],
                        columns=list(np.arange(0, 138) * 4 + 450) + [hyper_imges[0].target_varible_name])
    

def get_df_2_pca(hyper_imges: tp.Sequence[HyperImg], 
                 filter_value: str = '') -> pd.DataFrame:
    """
    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        filter_value (str, optional): if target varible = filter_value, that this sample is skipped. Defaults to ''.

    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.
    Returns:
        pd.DataFrame: DataFrame of 2 main PCA components.
    """
    
    if len(hyper_imges) == 0:
        raise EmptyHyperImgList
    
    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass
    
    pipe = Pipeline([('scaler', StandardScaler()), 
                     ('pca', PCA(n_components=2))])
    df = get_df_medians(hyper_imges)
    X = df.drop([hyper_imges[0].target_varible_name], axis=1)
    y = df[[hyper_imges[0].target_varible_name]]
    X = pipe.fit_transform(X)
    new_arr = list(zip(X[:, :2], y[hyper_imges[0].target_varible_name]))
    lst_of_value = [(new_arr[i][0][0], new_arr[i][0][1], new_arr[i][1])
                    for i, _ in enumerate(new_arr)
                    if new_arr[i][1] != filter_value]
    return pd.DataFrame(lst_of_value, columns=['1', '2', 
                                               hyper_imges[0].target_varible_name])
