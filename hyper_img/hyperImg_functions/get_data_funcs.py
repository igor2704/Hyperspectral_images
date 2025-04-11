from hyper_img.hyperImg import HyperImg
from hyper_img.hyperImg_segmenter.segmenter import Segmenter
from hyper_img.hyperImg_data_classes.data import HyperData

import typing as tp

import os
import numpy as np
import pandas as pd
import tifffile as tiff

import scipy.stats
from scipy.stats import mannwhitneyu, ttest_ind

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from collections import defaultdict, Counter
from copy import deepcopy
from itertools import combinations

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


def get_df_graphics_features_wavelenght(hyper_data: HyperData) -> pd.DataFrame:
    """
    Create DataFrame for graphics features versus wavelength.
    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.

    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.

    Returns:
        pd.DataFrame: DataFrame for graphics medians versus wavelength.
    """
    x_axis: tp.List[int] = list(np.arange(0, hyper_data.channels_len) * hyper_data.camera_sensitive
                                + hyper_data.camera_begin_wavelenght)
    points: tp.List[tp.Any] = list()

    wl_columns = ['wl_' + str(wl) 
                  for wl in np.arange(0, hyper_data.channels_len) * hyper_data.camera_sensitive + hyper_data.camera_begin_wavelenght]
    
    for sample_number in range(len(hyper_data.hyper_table)):
        feature_df = hyper_data.hyper_table.iloc[sample_number]
        feature = feature_df[wl_columns]
        feature = feature.to_numpy()
        object_name = feature_df['Object name']
        factors = feature_df['all_factors']
        for p in zip(x_axis, feature):
            points.append([p[0], p[1], sample_number, object_name, factors])

    return pd.DataFrame(points, columns=['Wavelength', hyper_data.feature_name,
                                         'Sample', 'Object name', 'all_factors'])


def get_boxplot_values(hyper_data: HyperData,
                       channels: tp.Sequence[int] | None = None) -> dict[int]:
    """
    Create data for boxplots.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        channels (tp.Sequence[int] | None, optional):channels used to build boxplots charts. Defaults to None.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.

    Returns:
        dict[int]: pixel value dictionary.
    """

    all_channels = channels if channels is not None else range(hyper_data.channels_len)

    values = defaultdict(dict)
    target_variable_names = set()
    for j, img in enumerate(hyper_data.hyper_imgs_lst):
        target_variable_names.add(img.get_factors())
        all_values = img.get_all_values()
        for i in all_channels:
            factors = ''.join([f'{k}:{img.factors[k]};' for k in img.factors])
            values[i][(factors + img.object_name, 
                       img.get_factors(), j)] = all_values[i]

    return dict(values)


def get_df_pca_and_explained_variance(hyper_data: HyperData,
                                      n_components: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Create pd.DataFrame with projection values on n_components main vectors in PCA, and explained variance.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        n_components (int): space dimension after downscaling.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.

    Returns:
        pd.DataFrame: DataFrame of 2 main PCA components.
        np.ndarray: numpy array with explained variance for n_components main PCA components.
    """
    pipe = Pipeline([('scaler', StandardScaler(with_std=False)),
                     ('pca', PCA(n_components=n_components))])
    df = deepcopy(hyper_data.hyper_table)
    X = df.drop(['Object name', 'all_factors'] + hyper_data.factors, axis=1)
    X = pipe.fit_transform(X)

    lst_of_value = [list(X[i]) + [df['Object name'][i], df['all_factors'][i]]
                    for i, _ in enumerate(X)]
    columns_numbers = [str(i + 1) for i in range(n_components)]
    return pd.DataFrame(lst_of_value,
                        columns=columns_numbers + ['Object name', 'all_factors']), \
           pipe['pca'].explained_variance_ratio_[:n_components]


def get_df_isomap(hyper_data: HyperData,
		          n_neighbors: int  = 5,
                  n_components: int = 2,
                  **kwargs_sklearn_isomap) -> pd.DataFrame:
    """
    Create pd.DataFrame with projection values on 2 main vectors in ISOMAP

    Args:
    	n_neighbors (int): number of neighbors to consider for each point. Default 5.
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        n_components (int): space dimension after downscaling.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.

    Returns:
        pd.DataFrame: DataFrame of 2 main ISOMAP components.
    """ 
    pipe = Pipeline([('scaler', StandardScaler(with_std=False)),
                     ('isomap', Isomap(n_neighbors=n_neighbors, n_components=n_components, **kwargs_sklearn_isomap))])
    df = deepcopy(hyper_data.hyper_table)
    X = df.drop(['Object name', 'all_factors'] + hyper_data.factors, axis=1)
    X = pipe.fit_transform(X)

    lst_of_value = [list(X[i]) + [df['Object name'][i], df['all_factors'][i]]
                    for i, _ in enumerate(X)]
    columns_numbers = [str(i + 1) for i in range(n_components)]
    return pd.DataFrame(lst_of_value,
                        columns=columns_numbers + ['Object name', 'all_factors'])


def get_df_umap(hyper_data: HyperData,
                n_components: int = 2,
                n_neighbors: int = 15,
                **kwargs_sklearn_umap) -> pd.DataFrame:
    """
    Create pd.DataFrame with projection values on 2 main vectors in UMAP

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        n_neighbors (int | None): number of neighbors to consider for each point. Default 15.
        n_components (int): space dimension after downscaling.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
    Raises:
        NeedHyperImgSubclass: the image class must be a subclass of HyperImg .
        EmptyHyperImgList: the sequence of images was empty.

    Returns:
        pd.DataFrame: DataFrame of 2 main UMAP components.
    """
    df = deepcopy(hyper_data.hyper_table)
    X = df.drop(['Object name', 'all_factors'] + hyper_data.factors, axis=1)

    pipe = make_pipeline(SimpleImputer(strategy='mean'), PowerTransformer())
    X = pipe.fit_transform(X)
    y = df[['all_factors']]
    y, _ = pd.factorize(y['all_factors'])
    
    manifold = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, **kwargs_sklearn_umap).fit(X, y)
    X = manifold.transform(X)

    lst_of_value = [list(X[i]) + [df['Object name'][i], df['all_factors'][i]]
                    for i, _ in enumerate(X)]
    columns_numbers = [str(i + 1) for i in range(n_components)]
    return pd.DataFrame(lst_of_value,
                        columns=columns_numbers + ['Object name', 'all_factors'])


def get_mannwhitneyu_pairwise(hyper_data,
                       alpha=0.05,
                       factors=None,
                       alternative: str = 'two-sided',
                       corrected_p_value_method: str | None = 'holm',
                       params_scipy: dict[str, tp.Any] | None = None):
    df = deepcopy(hyper_data.hyper_table)
    factors_lst = list()
    if factors is not None:
        for i in range(len(df)):
            value = ''
            for f in factors:
                value += f'{f}_{str(df.iloc[i][f])}; '
            factors_lst.append(value[:-2])
    else:
        factors_lst = list(df['all_factors'])
    df['fffactors'] = factors_lst
    
    if params_scipy is None:
        params_scipy = dict()
    
    dct_answer = defaultdict(list)
    for ch in range(hyper_data.channels_len):
        wl = ch * hyper_data.camera_sensitive + hyper_data.camera_begin_wavelenght
        dct = dict()
        pvalues = list()
        for i, (f1, f2) in enumerate(combinations(df['fffactors'].unique(), 2)):
            arr_hyper_1 = df[df['fffactors'] == f1][f'wl_{str(wl)}'].to_numpy()
            arr_hyper_2 = df[df['fffactors'] == f2][f'wl_{str(wl)}'].to_numpy()

            p_value = mannwhitneyu(arr_hyper_1, arr_hyper_2, alternative=alternative, **params_scipy)[1]
            dct[i] = f'{f1} : {f2}'
            pvalues.append(p_value)
        
        for k in dct:
            dct_answer[dct[k]].append(pvalues[k])
    
    value_idx = dict()
    for i, k in enumerate(dct_answer):
        value_idx[k] = i
    
    if corrected_p_value_method is not None:
        all_p_value = []
        lens = dict()
        lens[-1] = 0
        for k in dct_answer:
            all_p_value.extend(dct_answer[k])
            lens[value_idx[k]] = len(dct_answer[k]) + lens[value_idx[k] - 1]
        all_p_value = multipletests(all_p_value, method=corrected_p_value_method, alpha=alpha)[1]
        for k in dct_answer:
            dct_answer[k] = all_p_value[lens[value_idx[k] - 1]:lens[value_idx[k]]]
    
    wl_columns = ['wl_' + str(wl) 
                  for wl in np.arange(0, hyper_data.channels_len) * hyper_data.camera_sensitive + hyper_data.camera_begin_wavelenght]
    return pd.DataFrame(dct_answer, index=wl_columns)


def get_ttest_pairwise(hyper_data,
                       alpha=0.05,
                       factors=None,
                       alternative: str = 'two-sided',
                       corrected_p_value_method: str | None = 'holm',
                       params_scipy: dict[str, tp.Any] | None = None):
    df = deepcopy(hyper_data.hyper_table)
    factors_lst = list()
    if factors is not None:
        for i in range(len(df)):
            value = ''
            for f in factors:
                value += f'{f}_{str(df.iloc[i][f])}; '
            factors_lst.append(value[:-2])
    else:
        factors_lst = list(df['all_factors'])
    df['fffactors'] = factors_lst
    
    if params_scipy is None:
        params_scipy = dict()
    
    dct_answer = defaultdict(list)
    for ch in range(hyper_data.channels_len):
        wl = ch * hyper_data.camera_sensitive + hyper_data.camera_begin_wavelenght
        dct = dict()
        pvalues = list()
        for i, (f1, f2) in enumerate(combinations(df['fffactors'].unique(), 2)):
            arr_hyper_1 = df[df['fffactors'] == f1][f'wl_{str(wl)}'].to_numpy()
            arr_hyper_2 = df[df['fffactors'] == f2][f'wl_{str(wl)}'].to_numpy()

            p_value = ttest_ind(arr_hyper_1, arr_hyper_2, alternative=alternative, **params_scipy)[1]
            dct[i] = f'{f1} : {f2}'
            pvalues.append(p_value)
        
        pvalues = multipletests(pvalues, method=corrected_p_value_method, alpha=alpha)[1]
        for k in dct:
            dct_answer[dct[k]].append(pvalues[k])
    
    value_idx = dict()
    for i, k in enumerate(dct_answer):
        value_idx[k] = i
    
    if corrected_p_value_method is not None:
        all_p_value = []
        lens = dict()
        lens[-1] = 0
        for k in dct_answer:
            all_p_value.extend(dct_answer[k])
            lens[value_idx[k]] = len(dct_answer[k]) + lens[value_idx[k] - 1]
        all_p_value = multipletests(all_p_value, method=corrected_p_value_method, alpha=alpha)[1]
        for k in dct_answer:
            dct_answer[k] = all_p_value[lens[value_idx[k] - 1]:lens[value_idx[k]]]
    
    wl_columns = ['wl_' + str(wl) 
                  for wl in np.arange(0, hyper_data.channels_len) * hyper_data.camera_sensitive + hyper_data.camera_begin_wavelenght]
    return pd.DataFrame(dct_answer, index=wl_columns)


def get_tukey_pairwise(hyper_data,
                       alpha=0.05,
                       factors=None,
                       corrected_p_value_method: str | None = 'holm'):
    df = deepcopy(hyper_data.hyper_table)
    factors_lst = list()
    if factors is not None:
        for i in range(len(df)):
            value = ''
            for f in factors:
                value += f'{f} : {str(df.iloc[i][f])}; '
            factors_lst.append(value)
    else:
        factors_lst = list(df['all_factors'])
    df['fffactors'] = factors_lst
    
    dct_answer = defaultdict(list)
    for k in dct_answer:
        print(k)
    for ch in range(hyper_data.channels_len):
        wl = ch * hyper_data.camera_sensitive + hyper_data.camera_begin_wavelenght
        tukey = pairwise_tukeyhsd(endog=df[f'wl_{str(wl)}'], groups=df['fffactors'], alpha=0.05)
        tukey = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        for j in range(len(tukey)):
            dct_answer[f'{tukey.iloc[j]["group1"]} : {tukey.iloc[j]["group2"]}'].append(tukey.iloc[j]['p-adj'])
    
    value_idx = dict()
    for i, k in enumerate(dct_answer):
        value_idx[k] = i
    
    if corrected_p_value_method is not None:
        all_p_value = []
        lens = dict()
        lens[-1] = 0
        for k in dct_answer:
            all_p_value.extend(dct_answer[k])
            lens[value_idx[k]] = len(dct_answer[k]) + lens[value_idx[k] - 1]
        all_p_value = multipletests(all_p_value, method=corrected_p_value_method, alpha=alpha)[1]
        for k in dct_answer:
            dct_answer[k] = all_p_value[lens[value_idx[k] - 1]:lens[value_idx[k]]]

    wl_columns = ['wl_' + str(wl) 
                  for wl in np.arange(0, hyper_data.channels_len) * hyper_data.camera_sensitive + hyper_data.camera_begin_wavelenght]
    return pd.DataFrame(dct_answer, index=wl_columns)


def get_df_em_algorithm_clustering(hyper_data: HyperData,
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
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
        **kwargs_sklearn_gaussian_mixture: other params for sklearn.mixture.GaussianMixture.

    Returns:
        pd.DataFrame: result of clusterization.
        pd.DataFrame: table with brief description of clusters.
    """
    if downscaling_method == 'PCA':
        df, _ = get_df_pca_and_explained_variance(hyper_data, n_components=dim_clusterization)
    elif downscaling_method == 'ISOMAP':
        df = get_df_isomap(hyper_data, n_components=dim_clusterization, n_neighbors=n_neighbors_isomap)
    elif downscaling_method == 'UMAP':
        df = get_df_umap(hyper_data, n_components=dim_clusterization, n_neighbors=n_neighbors_umap)
    else:
        raise ThisMethodDoesNotExist('Please choose "PCA", "UMAP" or "ISOMAP" ')

    X = df.drop(['Object name', 'all_factors'], axis=1)
    y = df['all_factors']

    if n_clusters is None:
        n_clusters = len(y.unique())
    gm = GaussianMixture(n_components=n_clusters, n_init=n_init, init_params=init_params,
                         **kwargs_sklearn_gaussian_mixture).fit(X)
    clusters = gm.predict(X)
    df['Clusters'] = pd.Series(clusters)

    cluster_stat = []
    for i in range(n_clusters):
        dct = dict(Counter(df[df['Clusters'] == i]['all_factors']))
        max_values = np.max(list(dct.values()))
        max_class = [k for k in dct.keys() if dct[k] == max_values][0]
        accuracy = max_values / len(df[df['Clusters'] == i])
        cluster_stat.append([str(i + 1), max_class, accuracy])

    return df, pd.DataFrame(cluster_stat, columns=['cluster', 'prevailing class', 'accuracy']) 


def get_anova_df(hyper_data,
                 alpha=0.05,
                 corrected_p_value_method: str | None = 'holm'):
    df = deepcopy(hyper_data.hyper_table)
    dct_ind_to_f = dict()
    dct_f_to_ind = dict()
    for f in hyper_data.factors:
        for i, val in enumerate(df[f].unique()):
            dct_ind_to_f[(f, i)] = val
            dct_f_to_ind[(f, val)] = i

    rename_f_dct = dict()
    rename_f_dct_rev = dict()
    factor_lst = list()
    for i, f in enumerate(hyper_data.factors):
        rename_f_dct[f] = f'fffactor{i}'
        rename_f_dct_rev[f'fffactor{i}'] = f
        factor_lst.append(f'fffactor{i}')

    df = df.rename(columns=rename_f_dct)
    for f in factor_lst:
        df[f] = df[f].apply(lambda x: str(dct_f_to_ind[(rename_f_dct_rev[f], x)]))

    df = df.drop(columns=['Object name', 'all_factors'])

    feature_names = []
    columns = [c for c in df.columns if 'fffactor' in c]
    for i in range(len(columns)):
        for col in combinations(columns, i + 1):
            if 'fffactor' in ' '.join(col):
                feature_names.append(':'.join([f'C({c})' for c in col]))

    dct_answer = defaultdict(list)

    for ch in range(hyper_data.channels_len):
        wl = ch * hyper_data.camera_sensitive + hyper_data.camera_begin_wavelenght
        other_wl = ['wl_' + str(w * hyper_data.camera_sensitive + hyper_data.camera_begin_wavelenght) 
                                        for w in np.arange(0, hyper_data.channels_len) if w != ch]
        model = ols(f"wl_{str(wl)} ~ " + "+".join(feature_names), data=df).fit(
            cov_type="HC3"
        )

        df_res = sm.stats.anova_lm(model, typ=2)
        index = [idx.replace('C(', '').replace(')', '') for idx in df_res.index]
        for k in rename_f_dct_rev:
            index = [idx.replace(k, rename_f_dct_rev[k]) for idx in index]
        df_res.index = index
        for idx in df_res.index:
            if idx != 'Residual':
                dct_answer[idx].append(df_res.loc[idx]['PR(>F)'])
    
    value_idx = dict()
    for i, k in enumerate(dct_answer):
        value_idx[k] = i
    
    if corrected_p_value_method is not None:
        all_p_value = []
        lens = dict()
        lens[-1] = 0
        for k in dct_answer:
            all_p_value.extend(dct_answer[k])
            lens[value_idx[k]] = len(dct_answer[k]) + lens[value_idx[k] - 1]
        all_p_value = multipletests(all_p_value, method=corrected_p_value_method, alpha=alpha)[1]
        for k in dct_answer:
            dct_answer[k] = all_p_value[lens[value_idx[k] - 1]:lens[value_idx[k]]]
    wl_columns = ['wl_' + str(wl) 
                        for wl in np.arange(0, hyper_data.channels_len) * hyper_data.camera_sensitive + hyper_data.camera_begin_wavelenght]
    return pd.DataFrame(dct_answer, index=wl_columns)


def get_stat_tests_df(hyper_data,
                   alpha=0.05,
                   stat_test_names: tp.Sequence[str] = ('Tukey', 'Mann-Whitneyu', 'ANOVA'),
                   alternative: str = 'two-sided',
                   factors=None,
                   corrected_p_value_method: str | None = 'holm',
                   **kwargs):
    # stat_test_names subsequence of ['Tukey', 'Mann-Whitneyu', 'ttest', 'ANOVA']
    dct = defaultdict(list)
    for name in stat_test_names:
        if name == 'Tukey':
            df = get_tukey_pairwise(hyper_data, alpha, factors, corrected_p_value_method=None)
        elif name == 'Mann-Whitneyu':
            df = get_mannwhitneyu_pairwise(hyper_data, alpha, factors, 
                                            alternative, None, None)
        elif name == 'ttest':
            df = get_ttest_pairwise(hyper_data, alpha, factors, 
                                    alternative, None, None)
        elif name == 'ANOVA':
            df = get_anova_df(hyper_data, alpha, corrected_p_value_method=corrected_p_value_method)
        for col in df.columns:
            for i in range(len(df)):
                dct['Wavelength'].append(df.index[i].replace('wl_', ''))
                dct['P-value'].append(df.iloc[i][col])
                dct['Factor'].append(col)
                dct['Test name'].append(name)
    
    if corrected_p_value_method is not None:
        dct['P-value'] = multipletests(dct['P-value'], method=corrected_p_value_method, alpha=alpha)[1]
    
    df = pd.DataFrame(dct)
    return df
  
