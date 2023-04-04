from hyper_img.hyperImg import HyperImg
from hyper_img.hyperImg_functions.get_data_funcs import get_mean_diff_and_confident_interval_df, \
                                                        get_df_graphics_medians_wavelenght, \
                                                        get_df_2_pca_and_explained_variance, get_df_medians

import typing as tp

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as go
import plotly.express as px

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer

import datashader as ds
import datashader.transfer_functions as tf
import datashader.bundling as bd
import matplotlib.pyplot as plt
import colorcet
import matplotlib.colors
import matplotlib.cm
import bokeh.plotting as bpl
import bokeh.transform as btr
import holoviews as hv
import holoviews.operation.datashader as hd

import umap
import umap.plot

import warnings


def get_mean_diff_and_confident_interval_graph(hyper_imges: tp.Sequence[HyperImg],
                                               target_variable_1: str,
                                               target_variable_2: str,
                                               level: float = 0.95,
                                               download_path: str = '') -> None:
    """
    Plots confidence interval for sample mean differences.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        target_variable_1 (str): first target vaiable name.
        target_variable_2 (str): second target vaiable name.
        level (float): level for confident interval. Default 0.95 (95%).
        download_path (str): if not equal to '', then save the graph as download_path.
    """

    warnings.filterwarnings('ignore')

    m, l, r = get_mean_diff_and_confident_interval_df(hyper_imges, target_variable_1, target_variable_2, level)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=l['Wavelength'], y=l['Median'],
                             name='left border', line={'color': 'red'}))
    fig.add_trace(go.Scatter(x=r['Wavelength'], y=r['Median'], fill='tonexty',
                             name='right border', line={'color': 'red'}))
    fig.add_trace(go.Scatter(x=m['Wavelength'], y=m['Median'], name='difference between sample means',
                             line={'color': 'blue'}))

    fig.update_layout(title=f'{level * 100}% confident interval for difference between sample means '
                            f'({target_variable_1}, {target_variable_2})',
                      xaxis_title='Wavelength',
                      yaxis_title='Median')
    fig.show()

    if download_path:
        fig.write_html(download_path)


def get_medians_wavelenght_graph(hyper_imges: tp.Sequence[HyperImg],
                                 color: dict[str, str] | None = None,
                                 download_path: str = '',
                                 camera_sensitive: int = 4,
                                 camera_begin_wavelenght: int = 450,
                                 **kwargs) -> None:
    """
    Plots medians versus wavelength.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        color (dict[str, str] | None): dict with target variable colors. If None than use default colors. Default None.
        download_path (str): if not equal to '', then save the graph as download_path. Default ''.
        camera_sensitive (int): camera sensitive. Default 4.
        camera_begin_wavelenght (int): minimum wavelenght that camera can detect. Default 450.
    """

    df = get_df_graphics_medians_wavelenght(hyper_imges, camera_sensitive, camera_begin_wavelenght)

    if color is not None:
        fig = px.line(df, x='Wavelength', y='Median', color=hyper_imges[0].target_varible_name,
                      color_discrete_map=color, line_group=df['Sample'], hover_name='PlantNumber',
                      **kwargs)
    else:
        fig = px.line(df, x='Wavelength', y='Median', color=hyper_imges[0].target_varible_name,
                      line_group=df['Sample'], **kwargs)

    fig.update_layout(title=f'Dependence of medina on wavelength for {hyper_imges[0].target_varible_name}',
                      xaxis_title='Wavelength',
                      yaxis_title='Median')
    fig.update_traces(line=dict(width=0.5))

    fig.show()

    if download_path:
        fig.write_html(download_path)


def get_2_pca_graph(hyper_imges: tp.Sequence[HyperImg],
                    color: dict[str, str] | None = None,
                    download_path: str = '',
                    camera_sensitive: int = 4,
                    camera_begin_wavelenght: int = 450) -> None:
    """
    Plots PSA with 2 principal components.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        color (dict[str, str] | None): dict with target variable colors. If None than use default colors. Default None.
        download_path (str): if not equal to '', then save the graph as download_path. Default ''.
        camera_sensitive (int): camera sensitive. Default 4.
        camera_begin_wavelenght (int): minimum wavelenght that camera can detect. Default 450.

    """

    df, var = get_df_2_pca_and_explained_variance(hyper_imges, camera_sensitive, camera_begin_wavelenght)

    if color is not None:
        fig = px.scatter(df, x='1', y='2', color=hyper_imges[0].target_varible_name, hover_name='PlantNumber',
                         color_discrete_map=color)
    else:
        fig = px.scatter(df, x='1', y='2', color=hyper_imges[0].target_varible_name, hover_name='PlantNumber')

    fig.update_layout(title=f'PCA',
                      xaxis_title=f'first main component (explained variance {var[0]: 0.4f})',
                      yaxis_title=f'second main component (explained variance {var[1]: 0.4f})')
    fig.show()

    if download_path:
        fig.write_html(download_path)


def get_umap_graph(hyper_imges: tp.Sequence[HyperImg],
                   color: dict[str, str] | None = None,
                   download_path: str = '',
                   camera_sensitive: int = 4,
                   camera_begin_wavelenght: int = 450,
                   **kwargs) -> None:
    """
    Plots UMAP.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        color (dict[str, str] | None): dict with target variable colors. If None than use default colors. Default None.
        download_path (str): if not equal to '', then save the graph as download_path. Default ''.
        camera_sensitive (int): camera sensitive. Default 4.
        camera_begin_wavelenght (int): minimum wavelenght that camera can detect. Default 450.

    """

    df = get_df_medians(hyper_imges, camera_sensitive=camera_sensitive,
                        camera_begin_wavelenght=camera_begin_wavelenght)
    X = df.drop([hyper_imges[0].target_varible_name, 'PlantNumber'], axis=1)
    y = df[[hyper_imges[0].target_varible_name]]
    y, _ = pd.factorize(y[hyper_imges[0].target_varible_name])

    pipe = make_pipeline(SimpleImputer(strategy='mean'), PowerTransformer())
    X = pipe.fit_transform(X)
    manifold = umap.UMAP(**kwargs).fit(X, y)
    X = manifold.transform(X)

    if color is not None:
        fig = px.scatter(df, x=X[:, 0], y=X[:, 1], color=hyper_imges[0].target_varible_name, hover_name='PlantNumber',
                         color_discrete_map=color)
    else:
        fig = px.scatter(df, x=X[:, 0], y=X[:, 1], color=hyper_imges[0].target_varible_name, hover_name='PlantNumber')

    fig.update_layout(title=f'UMAP',
                      xaxis_title=f'UMAP 1',
                      yaxis_title=f'UMAP 2')
    fig.show()

    if download_path:
        fig.write_html(download_path)
