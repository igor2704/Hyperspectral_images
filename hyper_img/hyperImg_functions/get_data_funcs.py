from hyper_img.hyperImg import HyperImg

import typing as tp

import os
import numpy as np
import pandas as pd

from collections import defaultdict

import gspread
from google.oauth2.service_account import Credentials

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class EmptyHyperImgList(Exception):
    pass


class NeedHyperImgSubclass(Exception):
    pass


class NeedSequenceOrPath(Exception):
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


def get_list_hyper_img(class_name: type,
                       path: str | None = None, 
                       seq_names: tp.Sequence | None = None,
                       filter: tp.Callable[[str], bool] | None = None,
                       *args, **kwargs) -> list[HyperImg]:
    """
    Create the list of hyperspectral images.
    Args:
        class_name (type): images class.
        path (str | None): images path. Defaults to None.
        seq_names (pd.DataFrame | None): sequence with paths. Defaults to None.
        filter (tp.Callable[[str], bool] | None, optional): filter function. Defaults to None.

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
            hyper_img: HyperImg = class_name(name, *args, **kwargs)
            if filter is None or filter(hyper_img.target_varible):
                lst.append(hyper_img) 
    
    if seq_names is not None:
        for name in seq_names:
            hyper_img = class_name(name, *args, **kwargs)
            if filter is None or filter(hyper_img.target_varible):
                lst.append(hyper_img)
    
    if len(lst) == 0:
        raise EmptyHyperImgList
     
    return lst


def get_google_table_sheets(sheet_url: str = '1-C3XlMbsvuBdVyGzQ6eeVoTBIg9Fb9k6zUQjbzjYCtw',
                            authenticate_key_path: str = 'annotation-hyperspectral-8e9249a95022.json') -> pd.DataFrame:
    """
    For read google table.
    Args:
        sheet_url (str): table sheet url.
        authenticate_key_path (str): path to authenticate key.

    Returns:
        pd.DataFrame: table for read.
    """

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    credentials = Credentials.from_service_account_file(
        authenticate_key_path,
        scopes=scopes
    )

    gc = gspread.authorize(credentials).open_by_key(sheet_url)
    values = gc.get_worksheet(0).get_all_values()

    df = pd.DataFrame(data=values[1:], columns=values[0])
    df.replace('', pd.NA)

    return df


def create_table_annotation_df(hyper_imges: tp.Sequence[HyperImg],
                               name: str = 'Image Name',
                               black_calibr_name: str = 'Black calibration data',
                               white_calibr_name: str = 'White calibration data') -> pd.DataFrame:
    """
    Create table for annotation.
    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        name (str): column title with image name.
        black_calibr_name (str): column title with black image for calibration name.
        white_calibr_name (str): column title with white image for calibration name.

    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.

    Returns:
        pd.DataFrame: created table
    """
    if len(hyper_imges) == 0:
        raise EmptyHyperImgList
    
    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass

    table_dct: dict[str, list[tp.Any]] = defaultdict(list)
    
    for h_img in hyper_imges:
        table_dct[name].append(h_img.path.split('/')[-1])
        table_dct[black_calibr_name].append(h_img.black_calibration_img_name)
        table_dct[white_calibr_name].append(h_img.white_calibration_img_name)
        table_dct[h_img.target_varible_name].append(h_img.target_varible)
    
    return pd.DataFrame(table_dct)


def get_df_graphics_medians_wavelenght(hyper_imges: tp.Sequence[HyperImg],
                                       camera_sensitive: int = 4,
                                       camera_begin_wavelenght: int = 450) -> pd.DataFrame:
    """
    Create DataFrame for graphics medians versus wavelength.
    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        camera_sensitive (int): camera sensitive. Default 4.
        camera_begin_wavelenght (int): minimum wavelenght that camera can detect.

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

    x_axis: tp.List[int] = list(np.arange(0, hyper_imges[0].medians.shape[0]) * camera_sensitive
                                + camera_begin_wavelenght)
    points: tp.List[tp.Any] = list()

    for sample_number, sample in enumerate(hyper_imges):
        for p in zip(x_axis, sample.medians):
            points.append([p[0], p[1], sample_number, sample.target_varible])

    return pd.DataFrame(points, columns=['Wavelength', 'Median',
                                         'Sample', hyper_imges[0].target_varible_name])


def get_df_medians(hyper_imges: tp.Sequence[HyperImg], 
                   filter_value: str = '',
                   camera_sensitive: int = 4,
                   camera_begin_wavelenght: int = 450) -> pd.DataFrame:
    """
    Create DataFrame of medians.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        filter_value (str, optional): if target varible = filter_value, that this sample is skipped. Defaults to ''.
        camera_sensitive (int): camera sensitive. Default 4.
        camera_begin_wavelenght (int): minimum wavelenght that camera can detect. Default 450.

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
                        columns=list(np.arange(0, hyper_imges[0].medians.shape[0]) * camera_sensitive +
                                                    camera_begin_wavelenght) + [hyper_imges[0].target_varible_name])
    

def get_df_2_pca_and_explained_variance(hyper_imges: tp.Sequence[HyperImg],
                                        camera_sensitive: int = 4,
                                        camera_begin_wavelenght: int = 450) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Create pd.DataFrame with projection values on 2 main vectors in PCA, and explained variance.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        camera_sensitive (int): camera sensitive. Default 4.
        camera_begin_wavelenght (int): minimum wavelenght that camera can detect. Default 450.

    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.

    Returns:
        pd.DataFrame: DataFrame of 2 main PCA components.
        np.ndarray: numpy array with explained variance for 2 main PCA components.
    """
    
    if len(hyper_imges) == 0:
        raise EmptyHyperImgList
    
    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass
    
    pipe = Pipeline([('scaler', StandardScaler(with_std=False)), 
                     ('pca', PCA(n_components=2))])
    df = get_df_medians(hyper_imges, camera_sensitive=camera_sensitive,
                        camera_begin_wavelenght=camera_begin_wavelenght)
    X = df.drop([hyper_imges[0].target_varible_name], axis=1)
    y = df[[hyper_imges[0].target_varible_name]]
    X = pipe.fit_transform(X)
    new_arr = list(zip(X[:, :2], y[hyper_imges[0].target_varible_name]))
    lst_of_value = [(new_arr[i][0][0], new_arr[i][0][1], new_arr[i][1])
                    for i, _ in enumerate(new_arr)]
    return pd.DataFrame(lst_of_value, columns=['1', '2', hyper_imges[0].target_varible_name]), \
                                               pipe['pca'].explained_variance_ratio_[:2]


def get_mean_diff_and_95_confident_interval_df(hyper_imges: tp.Sequence[HyperImg],
                                               target_variable_1: str,
                                               target_variable_2: str) -> tuple[pd.DataFrame,
                                                                                pd.DataFrame,
                                                                                pd.DataFrame]:
    """
    Creates pd.DataFrames with sample mean difference, left bound, and right bound 95% confidence interval.

     Args:
         hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
         target_variable_1 (str): first target vaiable name.
         target_variable_2 (str): second target vaiable name.

    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.

    Returns:
        pd.DataFrame: difference of sample means values.
        pd.DataFrame: left bound of 95% confident interval.
        pd.DataFrame: right bound of 95% confident interval.
    """

    if len(hyper_imges) == 0:
        raise EmptyHyperImgList

    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass

    hyper_imgs_1 = [img for img in hyper_imges if img.target_varible == target_variable_1]
    hyper_imgs_2 = [img for img in hyper_imges if img.target_varible == target_variable_2]
    df_medians_wavelen_1 = get_df_graphics_medians_wavelenght(hyper_imgs_1).\
                                            sort_values('Wavelength').drop('Sample', axis=1)
    df_medians_wavelen_2 = get_df_graphics_medians_wavelenght(hyper_imgs_2).\
                                            sort_values('Wavelength').drop('Sample', axis=1)
    df_medians_wavelen_1['Wavelength'] = df_medians_wavelen_1['Wavelength'].astype('int')
    df_medians_wavelen_2['Wavelength'] = df_medians_wavelen_2['Wavelength'].astype('int')
    mean_1 = df_medians_wavelen_1.groupby('Wavelength', as_index=False).mean()
    mean_2 = df_medians_wavelen_2.groupby('Wavelength', as_index=False).mean()
    std_mean_1 = df_medians_wavelen_1.groupby('Wavelength', as_index=False).std()
    std_mean_1['Median'] /= np.sqrt(len(hyper_imgs_1))
    std_mean_2 = df_medians_wavelen_2.groupby('Wavelength', as_index=False).std()
    std_mean_2['Median'] /= np.sqrt(len(hyper_imgs_2))
    mean = mean_1.copy()
    mean['Median'] -= mean_2['Median']
    std = std_mean_1.copy()
    std['Median'] = (std_mean_1['Median'] ** 2 + std_mean_2['Median'] ** 2) ** (1/2)
    left = std.copy()
    left['Median'] = mean['Median'] - 2 * std['Median']
    right = std.copy()
    right['Median'] = mean['Median'] + 2 * std['Median']
    return mean, left, right
