from hyper_img.hyperImg import HyperImg
from hyper_img.hyperImg_segmenter.segmenter import Segmenter

import typing as tp

import os
import numpy as np
import pandas as pd
import tifffile as tiff

import scipy.stats
from scipy.stats import chisquare, mannwhitneyu

from collections import defaultdict, Counter

import gspread
from google.oauth2.service_account import Credentials

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import Isomap

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer

from sklearn.mixture import GaussianMixture

import umap

import warnings


class EmptyHyperImgList(Exception):
    pass


class NeedHyperImgSubclass(Exception):
    pass


class NeedSequenceOrPath(Exception):
    pass


class ThisMethodDoesNotExist(Exception):
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
                       segmenter: Segmenter,
                       path: str | None = None, 
                       seq_names: tp.Sequence | None = None,
                       filter: tp.Callable[[str], bool] | None = None,
                       same_samples: tp.Sequence = [],
                       norm_seq_tg_name: tp.Sequence = [],
                       *args, **kwargs) -> list[HyperImg]:
    """
    Create the list of hyperspectral images.
    Args:
        class_name (type): images class.
        path (str | None): images path. Defaults to None.
        seq_names (pd.DataFrame | None): sequence with paths. Defaults to None.
        filter (tp.Callable[[str], bool] | None, optional): filter function. Defaults to None.
        same_samples (tp.Sequence): same samples target variable names.
        norm_seq_tg_name (tp.Sequence): target variable names for normalization.
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

    norm_dct: dict[str, list[np.ndarray]] = defaultdict(list)
    same_samples_medians = []
       
    lst: list['HyperImg'] = list()
    if path is not None:
        for name in all_tiff_cube_img(path):
            hyper_img: HyperImg = class_name(name, segmenter=segmenter, *args, **kwargs)
            if filter is None or filter(hyper_img.target_variable):
                lst.append(hyper_img)

    if seq_names is not None:
        for name in seq_names:
            hyper_img = class_name(name, segmenter=segmenter, *args, **kwargs)
            if filter is None or filter(hyper_img.target_variable):
                lst.append(hyper_img)

    for hyper_img in lst:
        if hyper_img.target_variable not in same_samples:
            continue
        same_samples_medians.append(hyper_img.medians)

    std_axis_0 = lambda x: np.sqrt((len(x)/(len(x) - 1)) * np.var(x, axis=0))

    if len(same_samples_medians) > 0:
        mean = np.array(same_samples_medians).mean(axis=0)
        std = std_axis_0(same_samples_medians) if len(same_samples_medians) > 1 else 1
        for i, img in enumerate(lst):
            lst[i] = (img - mean) / std

    for hyper_img in lst:
        if hyper_img.target_variable not in norm_seq_tg_name:
            continue
        for name in norm_seq_tg_name:
            if hyper_img.target_variable == name:
                norm_dct[name].append(hyper_img.medians)
    
    if len(lst) == 0:
        raise EmptyHyperImgList

    keys = []
    for key, value in norm_dct.items():
        keys.append(key)
        std = std_axis_0(value) if len(value) > 1 else 1
        mean = np.array(value).mean(axis=0)
        for i, img in enumerate(lst):
            lst[i] = (img - mean) / std
        for hyper_img in lst:
            if hyper_img.target_variable in keys:
                continue
            if hyper_img.target_variable not in norm_seq_tg_name:
                continue
            for name in norm_seq_tg_name:
                if hyper_img.target_variable == name:
                    norm_dct[name].append(hyper_img.medians)

    return lst


def get_google_table_sheets(sheet_url: str,
                            authenticate_key_path: str) -> pd.DataFrame:
    """
    For read google table.
    Args:
        sheet_url (str): table sheet url.
        authenticate_key_path (str): path to authenticate key (https://cloud.google.com/iam/docs/keys-create-delete).

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
                               object_name: str = 'PlantNumber',
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
        if h_img.object_name is not None:
            table_dct[object_name].append(h_img.object_name)
        table_dct[black_calibr_name].append(h_img.black_calibration_img_name)
        table_dct[white_calibr_name].append(h_img.white_calibration_img_name)
        table_dct[h_img.target_varible_name].append(h_img.target_variable)
    
    return pd.DataFrame(table_dct)


def get_df_graphics_medians_wavelenght(hyper_imges: tp.Sequence[HyperImg]) -> pd.DataFrame:
    """
    Create DataFrame for graphics medians versus wavelength.
    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.

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

    x_axis: tp.List[int] = list(np.arange(0, hyper_imges[0].medians.shape[0]) * hyper_imges[0].camera_sensitive
                                + hyper_imges[0].camera_begin_wavelenght)
    points: tp.List[tp.Any] = list()

    for sample_number, sample in enumerate(hyper_imges):
        for p in zip(x_axis, sample.medians):
            points.append([p[0], p[1], sample_number, sample.target_variable, sample.object_name])

    return pd.DataFrame(points, columns=['Wavelength', 'Median',
                                         'Sample', hyper_imges[0].target_varible_name, 'Object name'])


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

    return pd.DataFrame([list(sample.medians) + [sample.target_variable, sample.object_name] for sample in hyper_imges
                         if sample.target_variable != filter_value],
                        columns=list(np.arange(0, hyper_imges[0].medians.shape[0]) * hyper_imges[0].camera_sensitive +
                                                  hyper_imges[0].camera_begin_wavelenght)
                                + [hyper_imges[0].target_varible_name, 'Object name'])
    

def get_df_pca_and_explained_variance(hyper_imges: tp.Sequence[HyperImg],
                                      n_components: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Create pd.DataFrame with projection values on n_components main vectors in PCA, and explained variance.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        n_components (int): space dimension after downscaling.
    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.

    Returns:
        pd.DataFrame: DataFrame of 2 main PCA components.
        np.ndarray: numpy array with explained variance for n_components main PCA components.
    """
    
    if len(hyper_imges) == 0:
        raise EmptyHyperImgList
    
    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass
    
    pipe = Pipeline([('scaler', StandardScaler(with_std=False)), 
                     ('pca', PCA(n_components=n_components))])
    df = get_df_medians(hyper_imges)
    X = df.drop([hyper_imges[0].target_varible_name, 'Object name'], axis=1)
    y = df[[hyper_imges[0].target_varible_name]]
    X = pipe.fit_transform(X)

    new_arr = list(zip(X, y[hyper_imges[0].target_varible_name]))
    lst_of_value = [list(new_arr[i][0][:]) + [new_arr[i][1]] + [df['Object name'][i]] for i, _ in enumerate(new_arr)]
    columns_numbers = [str(i + 1) for i in range(n_components)]
    return pd.DataFrame(lst_of_value,
                        columns=columns_numbers + [hyper_imges[0].target_varible_name, 'Object name']), \
           pipe['pca'].explained_variance_ratio_[:n_components]


def get_df_isomap(hyper_imges: tp.Sequence[HyperImg],
		          n_neighbors: int  = 5,
                  n_components: int = 2,
                  **kwargs_sklearn_isomap) -> pd.DataFrame:
    """
    Create pd.DataFrame with projection values on 2 main vectors in ISOMAP

    Args:
    	n_neighbors (int): number of neighbors to consider for each point. Default 5.
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        n_components (int): space dimension after downscaling.
    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.

    Returns:
        pd.DataFrame: DataFrame of 2 main ISOMAP components.
    """

    if len(hyper_imges) == 0:
        raise EmptyHyperImgList

    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass

    pipe = Pipeline([('scaler', StandardScaler(with_std=False)),
                     ('isomap', Isomap(n_neighbors=n_neighbors, n_components=n_components, **kwargs_sklearn_isomap))])
    df = get_df_medians(hyper_imges)
    X = df.drop([hyper_imges[0].target_varible_name, 'Object name'], axis=1)
    y = df[[hyper_imges[0].target_varible_name]]
    X = pipe.fit_transform(X)

    new_arr = list(zip(X, y[hyper_imges[0].target_varible_name]))
    lst_of_value = [list(new_arr[i][0][:]) + [new_arr[i][1]] + [df['Object name'][i]] for i, _ in enumerate(new_arr)]
    columns_numbers = [str(i + 1) for i in range(n_components)]
    return pd.DataFrame(lst_of_value,
                        columns=columns_numbers + [hyper_imges[0].target_varible_name, 'Object name'])


def get_df_umap(hyper_imges: tp.Sequence[HyperImg],
                n_components: int = 2,
                n_neighbors: int = 15,
                **kwargs_sklearn_umap) -> pd.DataFrame:
    """
    Create pd.DataFrame with projection values on 2 main vectors in UMAP

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        n_neighbors (int | None): number of neighbors to consider for each point. Default 15.
        n_components (int): space dimension after downscaling.
    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.

    Returns:
        pd.DataFrame: DataFrame of 2 main UMAP components.
    """

    if len(hyper_imges) == 0:
        raise EmptyHyperImgList

    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass

    df = get_df_medians(hyper_imges)
    X = df.drop([hyper_imges[0].target_varible_name, 'Object name'], axis=1)
    y = df[[hyper_imges[0].target_varible_name]]
    y, _ = pd.factorize(y[hyper_imges[0].target_varible_name])

    pipe = make_pipeline(SimpleImputer(strategy='mean'), PowerTransformer())
    X = pipe.fit_transform(X)
    manifold = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, **kwargs_sklearn_umap).fit(X, y)
    X = manifold.transform(X)

    new_arr = list(zip(X, df[hyper_imges[0].target_varible_name]))
    lst_of_value = [list(new_arr[i][0][:]) + [new_arr[i][1]] + [df['Object name'][i]] for i, _ in enumerate(new_arr)]
    columns_numbers = [str(i + 1) for i in range(n_components)]
    return pd.DataFrame(lst_of_value,
                        columns=columns_numbers + [hyper_imges[0].target_varible_name, 'Object name'])


def get_mean_diff_and_confident_interval_df(hyper_imges: tp.Sequence[HyperImg],
                                               target_variable_1: str,
                                               target_variable_2: str,
                                               level: float = 0.95) -> tuple[pd.DataFrame,
                                                                             pd.DataFrame,
                                                                             pd.DataFrame]:
    """
    Creates pd.DataFrames with sample mean difference, left bound, and right bound 95% confidence interval.

     Args:
         hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
         target_variable_1 (str): first target vaiable name.
         target_variable_2 (str): second target vaiable name.
         level (float): level for confident interval. Default 0.95 (95%).
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

    alpha = scipy.stats.norm.ppf((1 - level)/2 + level)

    hyper_imgs_1 = [img for img in hyper_imges if img.target_variable == target_variable_1]
    hyper_imgs_2 = [img for img in hyper_imges if img.target_variable == target_variable_2]
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
    left['Median'] = mean['Median'] - alpha * std['Median']
    right = std.copy()
    right['Median'] = mean['Median'] + alpha * std['Median']
    return mean, left, right


def get_chi2_p_value_df(hyper_imges: tp.Sequence[HyperImg],
                        target_variable_1: str,
                        target_variable_2: str,
                        number_bins: int = 5) -> pd.DataFrame:

    """
    Chi-square test to test the hypothesis about the coincidence
    of the distributions of the two groups for each channel.

    Args:
         hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
         target_variable_1 (str): first target vaiable name.
         target_variable_2 (str): second target vaiable name.
         number_bins (int): number of bins. Defaults 5.

    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.

    Returns:
        pd.DataFrame: DataFrame with p-values for each channel.
    """
    if len(hyper_imges) == 0:
        raise EmptyHyperImgList

    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass

    arr_hyper_1 = np.hstack([img.get_medians().reshape(len(hyper_imges[0].medians), 1) for img in hyper_imges
                             if img.target_variable == target_variable_1])
    arr_hyper_2 = np.hstack([img.get_medians().reshape(len(hyper_imges[0].medians), 1) for img in hyper_imges
                             if img.target_variable == target_variable_2])

    assert len(arr_hyper_1) == len(arr_hyper_2) == len(hyper_imges[0].medians), len(arr_hyper_1)

    p_value = []
    for i in range(len(hyper_imges[0].medians)):
        arr_hyper_1_i = arr_hyper_1[i]
        arr_hyper_2_i = arr_hyper_2[i]

        min = np.min([np.min(arr_hyper_1_i), np.min(arr_hyper_2_i)]) - 10**(-9)
        max = np.max([np.max(arr_hyper_1_i), np.max(arr_hyper_2_i)]) + 10**(-9)
        len_interval = (max - min) / number_bins

        prob_1 = []
        prob_2 = []
        for j in range(number_bins):
            idx_1 = np.logical_and(min + j * len_interval <= arr_hyper_1_i,
                                   arr_hyper_1_i < min + (j + 1) * len_interval)
            prob_1.append(len(arr_hyper_1_i[idx_1]) / len(arr_hyper_1_i))
            idx_2 = np.logical_and(min + j * len_interval <= arr_hyper_2_i,
                                   arr_hyper_2_i < min + (j + 1) * len_interval)
            prob_2.append(len(arr_hyper_2_i[idx_2]) / len(arr_hyper_2_i))

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                p_value.append((i * hyper_imges[0].camera_sensitive + hyper_imges[0].camera_begin_wavelenght,
                                chisquare(prob_2, f_exp=prob_1)[1]))
            except RuntimeWarning:
                try:
                    p_value.append((i * hyper_imges[0].camera_sensitive + hyper_imges[0].camera_begin_wavelenght,
                                    chisquare(prob_1, f_exp=prob_2)[1]))
                except RuntimeWarning:
                    raise RuntimeWarning('please, decrease number of bins')

    return pd.DataFrame(p_value, columns=['wavelength, nm', 'p-value'])


def get_mannwhitneyu_p_value_df(hyper_imges: tp.Sequence[HyperImg],
                                target_variable_1: str,
                                target_variable_2: str,
                                alternative: str = 'two-sided',
                                params_scipy: dict[str, tp.Any] | None = None) -> pd.DataFrame:
    """
    Perform the Mann-Whitney U rank test on two groups.
    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        target_variable_1 (str): first target vaiable name.
        target_variable_2 (str): second target vaiable name.
        alternative (str): defines the alternative hypothesis ('two-sided', 'less', 'greater'). Default is ‘two-sided’
        params_scipy (dict[str, tp.Any] | None): dict with params for scipy.stats.mannwhitneyu.
                                                Default None (no extra params).
    Returns:
        pd.DataFrame: table with p-values.
    """
    if len(hyper_imges) == 0:
        raise EmptyHyperImgList

    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass

    if params_scipy is None:
        params_scipy = dict()

    arr_hyper_1 = np.hstack([img.get_medians().reshape(len(hyper_imges[0].medians), 1) for img in hyper_imges
                             if img.target_variable == target_variable_1])
    arr_hyper_2 = np.hstack([img.get_medians().reshape(len(hyper_imges[0].medians), 1) for img in hyper_imges
                             if img.target_variable == target_variable_2])

    assert len(arr_hyper_1) == len(arr_hyper_2) == len(hyper_imges[0].medians), len(arr_hyper_1)

    p_value = []
    for i in range(len(hyper_imges[0].medians)):
        arr_hyper_1_i = arr_hyper_1[i]
        arr_hyper_2_i = arr_hyper_2[i]

        p_value.append((i * hyper_imges[0].camera_sensitive + hyper_imges[0].camera_begin_wavelenght,
                        mannwhitneyu(arr_hyper_1_i, arr_hyper_2_i, alternative=alternative, **params_scipy)[1]))

    return pd.DataFrame(p_value, columns=['wavelength, nm', 'p-value'])


def get_df_em_algorithm_clustering(hyper_imges: tp.Sequence[HyperImg],
                                   filter: tp.Callable = lambda x: True,
                                   downscaling_method: str = 'PCA',
                                   dim_clusterization: int = 15,
                                   n_init: int = 30,
                                   n_clusters: int | None = None,
                                   init_params: str = 'random',
                                   n_neighbors_isomap: int = 5,
                                   n_neighbors_umap: int = 15,
                                   **kwargs_sklearn_gaussian_mixture) -> tuple[pd.DataFrame,
                                                                               pd.DataFrame]:
    """
    Clustering with the EM Algorith.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        filter (tp.Callable): function to filter a sequence of hyperspectral images.
        downscaling_method (str): downscaling method ('PCA', 'UMAP', 'ISOMAP'). Defaults 'PCA'.
        dim_clusterization (int): the dimension of the space in which clustering is performed. Defaults 15.
        n_init (int): the number of initializations to perform.
        n_clusters (int | None): the number of clusters. If None, then number of clusters == number of class
                                (unique target variable). Defaults None.
        init_params: the method used to initialize the weights, the means and the precisions
                     (‘kmeans’, ‘k-means++’, ‘random’, ‘random_from_data’). Defaults 'random'.
        n_neighbors_isomap (int): number of neighbors to consider for each point, if using isomap. Default 5.
        n_neighbors_umap (int): number of neighbors to consider for each point, if using umap. Default 15.
        **kwargs_sklearn_gaussian_mixture: other params for sklearn.mixture.GaussianMixture.

    Returns:
        pd.DataFrame: result of clusterization.
        pd.DataFrame: table with brief description of clusters.
    """
    if len(hyper_imges) == 0:
        raise EmptyHyperImgList

    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass

    filtred_hyper_imges = [img for img in hyper_imges if filter(img)]

    if downscaling_method == 'PCA':
        df, _ = get_df_pca_and_explained_variance(filtred_hyper_imges, n_components=dim_clusterization)
    elif downscaling_method == 'ISOMAP':
        df = get_df_isomap(filtred_hyper_imges, n_components=dim_clusterization, n_neighbors=n_neighbors_isomap)
    elif downscaling_method == 'UMAP':
        df = get_df_umap(filtred_hyper_imges, n_components=dim_clusterization, n_neighbors=n_neighbors_umap)
    else:
        raise ThisMethodDoesNotExist('Please choose "PCA", "UMAP" or "ISOMAP" ')

    X = df.drop([hyper_imges[0].target_varible_name, 'Object name'], axis=1)
    y = df[hyper_imges[0].target_varible_name]

    if n_clusters is None:
        n_clusters = len(y.unique())
    gm = GaussianMixture(n_components=n_clusters, n_init=n_init, init_params=init_params,
                         **kwargs_sklearn_gaussian_mixture).fit(X)
    clusters = gm.predict(X)
    df['Clusters'] = pd.Series(clusters)

    cluster_stat = []
    for i in range(n_clusters):
        dct = dict(Counter(df[df['Clusters'] == i][hyper_imges[0].target_varible_name]))
        max_values = np.max(list(dct.values()))
        max_class = [k for k in dct.keys() if dct[k] == max_values][0]
        accuracy = max_values / len(df[df['Clusters'] == i])
        cluster_stat.append([str(i + 1), max_class, accuracy])

    return df, pd.DataFrame(cluster_stat, columns=['cluster', 'prevailing class', 'accuracy'])
