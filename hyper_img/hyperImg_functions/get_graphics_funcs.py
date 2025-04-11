from hyper_img.hyperImg_data_classes.data import HyperData
from hyper_img.hyperImg_functions.get_data_funcs import get_df_graphics_features_wavelenght, get_boxplot_values, \
                                                        get_df_pca_and_explained_variance, get_df_isomap, \
                                                        get_df_umap, get_anova_df, get_df_em_algorithm_clustering, \
                                                        get_mannwhitneyu_pairwise, get_ttest_pairwise, get_tukey_pairwise, get_stat_tests_df

import os

import typing as tp

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio

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

import umap.plot

import warnings

from collections import defaultdict

from hyper_img.hyperImg_functions.plotly_color import plotly_color


class ThisMethodDoesNotExist(Exception):
    pass


def change_png_fig(fig):
    fig.update_layout(
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )


def get_boxplot_graphs(hyper_data: HyperData,
                       channels: tp.Sequence[int] | None = None,
                       color: dict[str, str] | None = None,
                       download_path_dir: str = '',
                       with_png: bool = False,
                       png_scale: float = 8,
                       png_width: int = 500,
                       png_height: int = 700,
                       fig_show: bool = False) -> None:
    """
    Plot boxplots.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        channels (tp.Sequence[int] | None, optional):channels used to build boxplots charts. Defaults to None (all channels).
        color (dict[str, str] | None): dict with target variable colors. If None than use default colors. Default None.
        download_path_dir (str, optional): name of the directory where the graph will be built (if not equal to ''). Defaults to ''.
        with_png (bool): if true, then save plots in png. Default False.
        png_scale (float): text scale in png plot. Default 5.
        png_width (int): width png plot. Default 950.
        png_height (int): height png plot. Default 700.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
        fig_show (bool): is it necessary to show graphs. Defaults to False.
    """
    values = get_boxplot_values(hyper_data, channels)

    all_channels = channels if channels is not None else range(hyper_data.channels_len)

    target_variable_names = set()
    for j, img in enumerate(hyper_data.hyper_imgs_lst):
        target_variable_names.add(img.get_factors())

    if color is None:
        colors = dict()
        for i, name in enumerate(target_variable_names):
            colors[name] = plotly_color[i]
    else:
        colors = color

    if download_path_dir:
        os.mkdir(download_path_dir)

    for ch in all_channels:
        trace_list = [go.Box(visible=True, y=values[ch][k], name=k[0] + f'  id: {k[2]}',
                        line=dict(color=colors[k[1]]), text=k[1])
                      for k in values[ch]]

        fig = go.Figure(data=trace_list)
        if fig_show:
            fig.show()

        if download_path_dir:
            path = os.path.join(download_path_dir,
                                f'boxplot_wavelength:{ch * hyper_data.camera_sensitive + hyper_data.camera_begin_wavelenght}.html')
            fig.write_html(path)
            if with_png:
                change_png_fig(fig)
                pio.write_image(fig, path.replace('.html', '.png'), scale=png_scale,
                                width=png_width, height=png_height)


def get_features_wavelenght_graph(hyper_data: HyperData,
                                 color: dict[str, str] | None = None,
                                 download_path: str = '',
                                 with_png: bool = False,
                                 png_scale: float = 8,
                                 png_width: int = 500,
                                 png_height: int = 700,
                                 fig_show: bool = False,
                                 **kwargs) -> None:
    """
    Plots medians versus wavelength.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        color (dict[str, str] | None): dict with target variable colors. If None than use default colors. Default None.
        download_path (str): if not equal to '', then save the graph as download_path. Default ''.
        with_png (bool): if true, then save plots in png. Default False.
        png_scale (float): text scale in png plot. Default 5.
        png_width (int): width png plot. Default 950.
        png_height (int): height png plot. Default 700.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
        fig_show (bool): is it necessary to show graphs. Defaults to False.
    """

    df = get_df_graphics_features_wavelenght(hyper_data)

    if color is not None:
        fig = px.line(df, x='Wavelength', y=f'{hyper_data.feature_name}', color=df['all_factors'],
                      color_discrete_map=color, line_group=df['Sample'], hover_name='Object name',
                      **kwargs)
    else:
        fig = px.line(df, x='Wavelength', y=f'{hyper_data.feature_name}', color=df['all_factors'],
                      line_group=df['Sample'], **kwargs)

    fig.update_layout(title=f'Dependence of {hyper_data.feature_name} on wavelength',
                      xaxis_title='Wavelength, nm',
                      yaxis_title='Reflection')
    fig.update_traces(line=dict(width=0.5))

    if fig_show:
        fig.show()

    if download_path:
        fig.write_html(download_path)
        if with_png:
            change_png_fig(fig)
            pio.write_image(fig, download_path.replace('.html', '.png'), scale=png_scale,
                            width=png_width, height=png_height)


def get_2_pca_graph(hyper_data: HyperData,
                    color: dict[str, str] | None = None,
                    download_path: str = '',
                    with_png: bool = False,
                    png_scale: float = 8,
                    png_width: int = 500,
                    png_height: int = 700,
                    fig_show: bool = False) -> None:
    """
    Plots PCA with 2 principal components.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        color (dict[str, str] | None): dict with target variable colors. If None than use default colors. Default None.
        download_path (str): if not equal to '', then save the graph as download_path. Default ''.
        with_png (bool): if true, then save plots in png. Default False.
        png_scale (float): text scale in png plot. Default 5.
        png_width (int): width png plot. Default 950.
        png_height (int): height png plot. Default 700.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
        fig_show (bool): is it necessary to show graphs. Defaults to False.
    """

    df, var = get_df_pca_and_explained_variance(hyper_data, 2)

    traces = []
    color_dct = dict()
    if color is None:
        for i, name in enumerate(df['all_factors'].unique()):
            color_dct[name] = plotly_color[i]
    else:
        color_dct = color
    df['group_name'] = df['all_factors'] + '_' + df['Object name']
    for group, data in df.groupby('group_name'):
        trace = go.Scatter(
            x=data['1'],
            y=data['2'],
            mode='lines+markers',
            line=dict(color=color_dct[data['all_factors'].iloc[0]],
                      width=0.3),
            name=data['all_factors'].iloc[0] + data['Object name'].iloc[0],
            marker=dict(symbol='circle'),
            text=data['Object name'].iloc[0],
            showlegend=True
        )
        traces.append(trace)

    fig = go.Figure(data=traces)

    fig.update_layout(title=f'PCA',
                      xaxis_title=f'first main component (explained variance {var[0]: 0.4f})',
                      yaxis_title=f'second main component (explained variance {var[1]: 0.4f})')
    if fig_show:
        fig.show()

    if download_path:
        fig.write_html(download_path)
        if with_png:
            change_png_fig(fig)
            pio.write_image(fig, download_path.replace('.html', '.png'), scale=png_scale,
                            width=png_width, height=png_height)


def get_2_isomap_graph(hyper_data: HyperData,
		               n_neighbors: int = 5,
                       color: dict[str, str] | None = None,
                       download_path: str = '',
                       with_png: bool = False,
                       png_scale: float = 8,
                       png_width: int = 500,
                       png_height: int = 700,
                       fig_show: bool = False,
                       **kwargs) -> None:
    """
    Plots ISOMAP with 2 principal components.
    Args:
        n_neighbors(int): number of neighbors to consider for each point. Default 5.
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        color (dict[str, str] | None): dict with target variable colors. If None than use default colors. Default None.
        download_path (str): if not equal to '', then save the graph as download_path. Default ''.
         with_png (bool): if true, then save plots in png. Default False.
        png_scale (float): text scale in png plot. Default 5.
        png_width (int): width png plot. Default 950.
        png_height (int): height png plot. Default 700.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
        fig_show (bool): is it necessary to show graphs. Defaults to False.
    """

    df = get_df_isomap(hyper_data, n_neighbors, **kwargs)

    traces = []
    color_dct = dict()
    if color is None:
        for i, name in enumerate(df['all_factors'].unique()):
            color_dct[name] = plotly_color[i]
    else:
        color_dct = color
    df['group_name'] = df['all_factors'] + '_' + df['Object name']
    for group, data in df.groupby('group_name'):
        trace = go.Scatter(
            x=data['1'],
            y=data['2'],
            mode='lines+markers',
            line=dict(color=color_dct[data['all_factors'].iloc[0]],
                      width=0.3),
            name=data['all_factors'].iloc[0] + data['Object name'].iloc[0],
            marker=dict(symbol='circle'),
            text=data['Object name'].iloc[0],
            showlegend=True
        )
        traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(title=f'ISOMAP',
                      xaxis_title=f'ISOMAP 1',
                      yaxis_title=f'ISOMAP 2')
    if fig_show:
        fig.show()

    if download_path:
        fig.write_html(download_path)
        if with_png:
            change_png_fig(fig)
            pio.write_image(fig, download_path.replace('.html', '.png'), scale=png_scale,
                            width=png_width, height=png_height)

def get_2_umap_graph(hyper_data: HyperData,
                     n_neighbors: int = 15,
                     color: dict[str, str] | None = None,
                     download_path: str = '',
                     with_png: bool = False,
                     png_scale: float = 8,
                     png_width: int = 500,
                     png_height: int = 700,
                     fig_show: bool = False,
                     **kwargs) -> None:
    """
    Plots UMAP.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        color (dict[str, str] | None): dict with target variable colors. If None than use default colors. Default None.
        download_path (str): if not equal to '', then save the graph as download_path. Default ''.
        with_png (bool): if true, then save plots in png. Default False.
        png_scale (float): text scale in png plot. Default 5.
        png_width (int): width png plot. Default 950.
        png_height (int): height png plot. Default 700.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
        fig_show (bool): is it necessary to show graphs. Defaults to False.
    """

    df = get_df_umap(hyper_data, n_neighbors, **kwargs)

    traces = []
    color_dct = dict()
    if color is None:
        for i, name in enumerate(df['all_factors'].unique()):
            color_dct[name] = plotly_color[i]
    else:
        color_dct = color
    df['group_name'] = df['all_factors'] + '_' + df['Object name']
    for group, data in df.groupby('group_name'):
        trace = go.Scatter(
            x=data['1'],
            y=data['2'],
            mode='lines+markers',
            line=dict(color=color_dct[data['all_factors'].iloc[0]],
                      width=0.3),
            name=data['all_factors'].iloc[0] + data['Object name'].iloc[0],
            marker=dict(symbol='circle'),
            text=data['Object name'].iloc[0],
            showlegend=True
        )
        traces.append(trace)

    fig = go.Figure(data=traces)

    fig.update_layout(title=f'UMAP',
                      xaxis_title=f'UMAP 1',
                      yaxis_title=f'UMAP 2')
    if fig_show:
        fig.show()

    if download_path:
        fig.write_html(download_path)
        if with_png:
            change_png_fig(fig)
            pio.write_image(fig, download_path.replace('.html', '.png'), scale=png_scale,
                            width=png_width, height=png_height)


def get_stat_test_graph(funcs,
                  stat_test_name: str,
                  hyper_data,
                  color: dict[str, str] | None = None,
                  download_path: str = '',
                  with_png: bool = False,
                  png_scale: float = 8,
                  png_width: int = 500,
                  png_height: int = 700,
                  fig_show: bool = False,
                  **kwargs):
    df = funcs(hyper_data)
    dct = defaultdict(list)
    for col in df.columns:
        for i in range(len(df)):
            dct['Wavelength'].append(df.index[i].replace('wl_', ''))
            dct['P-value'].append(df.iloc[i][col])
            dct['Factor'].append(col)

    df = pd.DataFrame(dct)
    if color is not None:
        fig = px.line(df, x='Wavelength', y=f'P-value', color='Factor',
                      color_discrete_map=color,
                      **kwargs)
    else:
        fig = px.line(df, x='Wavelength', y=f'P-value', color='Factor',
                      **kwargs)

    fig.update_layout(title=f'{stat_test_name} result',
                      xaxis_title='Wavelength, nm',
                      yaxis_title='p-value')
    fig.update_traces(line=dict(width=0.5))

    if fig_show:
        fig.show()

    if download_path:
        fig.write_html(download_path)
        if with_png:
            change_png_fig(fig)
            pio.write_image(fig, download_path.replace('.html', '.png'), scale=png_scale,
                            width=png_width, height=png_height)


def get_stat_tests_graph(hyper_data,
                   alpha=0.05,
                   stat_test_names: tp.Sequence[str] = ('Tukey', 'Mann-Whitneyu', 'ANOVA'),
                   alternative: str = 'two-sided',
                   factors=None,
                   corrected_p_value_method: str | None = 'holm',
                   color: dict[str, str] | None = None,
                   download_dir_path: str = '',
                   with_png: bool = False,
                   png_scale: float = 8,
                   png_width: int = 500,
                   png_height: int = 700,
                   fig_show: bool = False,
                   **kwargs):
    # stat_test_names subsequence of ['Tukey', 'Mann-Whitneyu', 'ttest', 'ANOVA']
    df = get_stat_tests_df(hyper_data, alpha, stat_test_names, alternative, factors, corrected_p_value_method, **kwargs)
    for name in stat_test_names:
        if color is not None:
            fig = px.line(df[df['Test name'] == name], x='Wavelength', y=f'P-value', color='Factor',
                        color_discrete_map=color)
        else:
            fig = px.line(df[df['Test name'] == name], x='Wavelength', y=f'P-value', color='Factor')

        fig.update_layout(title=f'{name} result',
                        xaxis_title='Wavelength, nm',
                        yaxis_title='p-value')
        fig.update_traces(line=dict(width=0.5))

        if fig_show:
            fig.show()

        if download_dir_path:
            path = os.path.join(download_dir_path, name + '.html')
            fig.write_html(path)
            if with_png:
                change_png_fig(fig)
                pio.write_image(fig, path.replace('.html', '.png'), scale=png_scale,
                                width=png_width, height=png_height)
             


def get_tukey_pairwise_graph(hyper_data,
                             alpha=0.05,
                             factors=None,
                             corrected_p_value_method: str | None = 'holm',
                             color: dict[str, str] | None = None,
                             download_path: str = '',
                             with_png: bool = False,
                             png_scale: float = 8,
                             png_width: int = 500,
                             png_height: int = 700,
                             fig_show: bool = False,
                             **kwargs):
    get_stat_test_graph(lambda x: get_tukey_pairwise(x, alpha, factors, corrected_p_value_method),
                        'Tukey',
                         hyper_data,
                         color,
                         download_path,
                         with_png,
                         png_scale,
                         png_width,
                         png_height,
                         fig_show,
                         **kwargs)
    
    
def get_mannwhitneyu_pairwise_graph(hyper_data,
                                    alpha=0.05,
                              factors=None,
                              alternative: str = 'two-sided',
                              corrected_p_value_method: str | None = 'holm',
                              params_scipy: dict[str, tp.Any] | None = None,
                              color: dict[str, str] | None = None,
                              download_path: str = '',
                              with_png: bool = False,
                              png_scale: float = 8,
                              png_width: int = 500,
                              png_height: int = 700,
                              fig_show: bool = False,
                              **kwargs):
    get_stat_test_graph(lambda x: get_mannwhitneyu_pairwise(hyper_data, alpha, factors, 
                                                            alternative, corrected_p_value_method,
                                                            params_scipy),
                        'Mann-Whitneyu',
                         hyper_data,
                         color,
                         download_path,
                         with_png,
                         png_scale,
                         png_width,
                         png_height,
                         fig_show,
                         **kwargs)


def get_ttest_pairwise_graph(hyper_data,
                             alpha=0.05,
                       factors=None,
                       alternative: str = 'two-sided',
                       corrected_p_value_method: str | None = 'holm',
                       params_scipy: dict[str, tp.Any] | None = None,
                       color: dict[str, str] | None = None,
                       download_path: str = '',
                       with_png: bool = False,
                       png_scale: float = 8,
                       png_width: int = 500,
                       png_height: int = 700,
                       fig_show: bool = False,
                       **kwargs):
    get_stat_test_graph(lambda x: get_ttest_pairwise(hyper_data, alpha, factors, 
                                                            alternative, corrected_p_value_method,
                                                            params_scipy),
                        'ttest',
                         hyper_data,
                         color,
                         download_path,
                         with_png,
                         png_scale,
                         png_width,
                         png_height,
                         fig_show,
                         **kwargs)


def get_anova_graph(hyper_data,
                    alpha=0.05,
                    corrected_p_value_method: str | None = 'holm',
                    color: dict[str, str] | None = None,
                    download_path: str = '',
                    with_png: bool = False,
                    png_scale: float = 8,
                    png_width: int = 500,
                    png_height: int = 700,
                    fig_show: bool = False,
                    **kwargs) -> None:
    get_stat_test_graph(lambda x: get_anova_df(x, alpha, corrected_p_value_method=corrected_p_value_method),
                        'ANOVA',
                        hyper_data,
                        color,
                        download_path,
                        with_png,
                        png_scale,
                        png_width,
                        png_height,
                        fig_show,
                        **kwargs)


def get_em_algorithm_clustering_graph(hyper_data: HyperData,
                                      downscaling_method: str = 'PCA',
                                      dim_clusterization: int = 15,
                                      n_clusters: int | None = None,
                                      n_neighbors_isomap: int = 5,
                                      n_neighbors_umap: int = 15,
                                      n_init: int = 30,
                                      init_params: str = 'random',
                                      color: dict[str, str] | None = None,
                                      download_path: str = '',
                                      download_path_table: str = '',
                                      with_png: bool = False,
                                      png_scale: float = 8,
                                      png_width: int = 500,
                                      png_height: int = 700,
                                      fig_show: bool = False,
                                      **kwargs_sklearn_gaussian_mixture) -> None:
    """
    Builds a graph in two-dimensional space after clustering in multi-dimensional space.
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
        color (dict[str, str] | None): dict with target variable colors. If None than use default colors. Default None.
        download_path (str): if not equal to '', then save the graph as download_path. Default ''.
        download_path_table (str): if not equal to '', then save the table as download_path. Default ''.
        with_png (bool): if true, then save plots in png. Default False.
        png_scale (float): text scale in png plot. Default 5.
        png_width (int): width png plot. Default 950.
        png_height (int): height png plot. Default 700.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
        fig_show (bool): is it necessary to show graphs. Defaults to False.
        **kwargs_sklearn_gaussian_mixture: other params for sklearn.mixture.GaussianMixture.
    """
    df, annotation = get_df_em_algorithm_clustering(hyper_data, downscaling_method, dim_clusterization,
                                                    n_init, n_clusters, init_params, n_neighbors_isomap,
                                                    n_neighbors_umap,
                                                    **kwargs_sklearn_gaussian_mixture)
    if downscaling_method == 'PCA':
        df_2d, _ = get_df_pca_and_explained_variance(hyper_data)
    elif downscaling_method == 'ISOMAP':
        df_2d = get_df_isomap(hyper_data, n_neighbors=n_neighbors_isomap)
    elif downscaling_method == 'UMAP':
        df_2d = get_df_umap(hyper_data, n_neighbors=n_neighbors_umap)
    else:
        raise ThisMethodDoesNotExist('Please choose "PCA", "UMAP" or "ISOMAP" ')

    df_2d['all_factors'] = pd.Series([tg_v + ' cluster: ' + str(cluster)
                                                           for tg_v, cluster in zip(df['all_factors'].values,
                                                                                    df['Clusters'])])

    if color is not None:
        fig = px.scatter(df_2d, x='1', y='2', color='all_factors', hover_name='Object name',
                         color_discrete_map=color)
    else:
        fig = px.scatter(df_2d, x='1', y='2', color='all_factors', hover_name='Object name')

    fig.update_layout(title=f'{downscaling_method}',
                      xaxis_title=f'{downscaling_method} 1',
                      yaxis_title=f'{downscaling_method} 2')
    if fig_show:
        fig.show()

    if download_path:
        fig.write_html(download_path)
        if with_png:
            change_png_fig(fig)
            pio.write_image(fig, download_path.replace('.html', '.png'), scale=png_scale,
                            width=png_width, height=png_height)
    if download_path_table:
        annotation.to_excel(download_path_table)


def create_folder_with_all_graphs(hyper_data: HyperData,
                                  alpha = 0.05,
                                  folder_path: str = 'all_hyper_graphics',
                                  stat_test_names:tp.Sequence[str] = ('Tukey', 'Mann-Whitneyu', 'ANOVA'),
                                  color: dict[str, str] | None = None,
                                  dim_clusterization: int = 10,
                                  n_clusters: int | None = None,
                                  n_neighbors_isomap: int = 5,
                                  n_neighbors_umap: int = 15,
                                  factors_pairwise_tests: None | list[str] = None,
                                  alternative: str = 'two-sided',
                                  params_scipy: dict[str, tp.Any] | None = None,
                                  corrected_p_value_method: str | None = 'holm',
                                  boxplots_channels: tp.Sequence[int] | None = None,
                                  with_png: bool = False,
                                  png_scale: float = 8,
                                  png_width: int = 500,
                                  png_height: int = 700,
                                  fig_show: bool = False) -> None:
    """
    Plot all graphics.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        folder_path (str): folder path. Defaults 'all_hyper_graphics'.
        color (dict[str, str] | None): dict with target variable colors. If None than use default colors. Default None.
        dim_clusterization (int): the dimension of the space in which clustering is performed. Defaults 15.
        n_neighbors_isomap (int): number of neighbors to consider for each point for isomap. Default 5.
        n_neighbors_umap (int): number of neighbors to consider for each point for umap. Default 15.
        n_clusters (int | None): the number of clusters. If None, then number of clusters == number of class
                                 (unique target variable). Defaults None.
        number_bins_chi2 (int): number of bins in the chi-square test. If division by zero occurs, then the number of
                                bins decrease as long as this exception/warnings persist. Defaults 7.
        confident_interval_level (float): level for sample mean difference confident interval. Default 0.95 (95%).
        mannwhitneyu_alternative (str): defines the alternative hypothesis ('two-sided', 'less', 'greater') for Mann-Whitney U rank test. Default is ‘two-sided’
        mannwhitneyu_scipy_params (dict[str, tp.Any] | None): dict with params for scipy.stats.mannwhitneyu. Default None (no extra params).
        ttest_scipy_params (dict[str, tp.Any] | None): defines the alternative hypothesis ('two-sided', 'less', 'greater') for ttest. Default is ‘two-sided’
        corrected_p_value_method (str | None): method from statsmodel multipletests. Default 'holm'.
        boxplots_channels (tp.Sequence[int] | None): channels used to build boxplots charts. Defaults to None (all channels).
        with_png (bool): if true, then save plots in png. Default False.
        png_scale (float): text scale in png plot. Default 5.
        png_width (int): width png plot. Default 950.
        png_height (int): height png plot. Default 700.
        mean_aggregate_dim_reduction (bool): whether to average samples by object_name for dimensionality reduction methods. Defaults to False.
        mean_aggregate_visualization (bool): whether to average samples by object_name for visualization. Defaults to False.
        mean_aggregate_stat_anal (bool): whether to average samples by object_name for statistical analysis. Defaults to False.
        fig_show (bool): is it necessary to show graphs. Defaults to False.
    """
    os.mkdir(folder_path)
    get_boxplot_graphs(hyper_data=hyper_data, color=color,
                       channels=boxplots_channels,
                       download_path_dir=os.path.join(folder_path, 'boxplots'),
                       with_png=with_png,
                       png_scale=png_scale,
                       png_width=png_width,
                       png_height=png_height,
                       fig_show=fig_show)
    
    get_features_wavelenght_graph(hyper_data=hyper_data, color=color,
                                 download_path=folder_path + '/features_wavelen.html',
                                 with_png=with_png, png_scale=png_scale,
                                 png_width=png_width, png_height=png_height,
                                 fig_show=fig_show)
    
    os.mkdir(folder_path + '/visualization')
    get_2_pca_graph(hyper_data=hyper_data, color=color, download_path=folder_path + '/visualization' + '/pca.html',
                    with_png=with_png, png_scale=png_scale,
                    png_width=png_width, png_height=png_height,
                    fig_show=fig_show)
    get_2_umap_graph(hyper_data=hyper_data, color=color, n_neighbors=n_neighbors_umap,
                     download_path=folder_path + '/visualization' + '/umap.html',
                     with_png=with_png, png_scale=png_scale,
                     png_width=png_width, png_height=png_height,
                     fig_show=fig_show)
    get_2_isomap_graph(hyper_data=hyper_data, n_neighbors=n_neighbors_isomap,
                       color=color, download_path=folder_path + '/visualization' + '/isomap.html',
                       with_png=with_png, png_scale=png_scale,
                       png_width=png_width, png_height=png_height,
                       fig_show=fig_show)
    
    os.mkdir(folder_path + '/clusterization')
    get_em_algorithm_clustering_graph(hyper_data=hyper_data, downscaling_method='PCA', dim_clusterization=dim_clusterization,
                                      n_clusters=n_clusters, init_params='random', color=color,
                                      download_path=folder_path + '/clusterization' + '/clusterization_pca_random.html',
                                      download_path_table=folder_path + '/clusterization' + '/clusterization_pca_random.xlsx',
                                      with_png=with_png, png_scale=png_scale,
                                      png_width=png_width, png_height=png_height,
                                      fig_show=fig_show)
    get_em_algorithm_clustering_graph(hyper_data=hyper_data, downscaling_method='PCA', dim_clusterization=dim_clusterization,
                                      n_clusters=n_clusters, init_params='kmeans', color=color,
                                      download_path=folder_path + '/clusterization' + '/clusterization_pca_kmeans.html',
                                      download_path_table=folder_path + '/clusterization' + '/clusterization_pca_kmeans.xlsx',
                                      with_png=with_png, png_scale=png_scale,
                                      png_width=png_width, png_height=png_height,
                                      fig_show=fig_show)
    get_em_algorithm_clustering_graph(hyper_data=hyper_data, downscaling_method='ISOMAP', dim_clusterization=dim_clusterization,
                                      n_clusters=n_clusters, init_params='random', color=color,
                                      download_path=folder_path + '/clusterization' + '/clusterization_isomap_random.html',
                                      download_path_table=folder_path + '/clusterization' + '/clusterization_isomap_random.xlsx',
                                      with_png=with_png, png_scale=png_scale, n_neighbors_isomap=n_neighbors_isomap,
                                      png_width=png_width, png_height=png_height,
                                      fig_show=fig_show)
    get_em_algorithm_clustering_graph(hyper_data=hyper_data, downscaling_method='ISOMAP', dim_clusterization=dim_clusterization,
                                      n_clusters=n_clusters, init_params='kmeans', color=color,
                                      download_path=folder_path + '/clusterization' + '/clusterization_isomap_kmeans.html',
                                      download_path_table=folder_path + '/clusterization' + '/clusterization_isomap_kmeans.xlsx',
                                      with_png=with_png, png_scale=png_scale, n_neighbors_isomap=n_neighbors_isomap,
                                      png_width=png_width, png_height=png_height,
                                      fig_show=fig_show)
    get_em_algorithm_clustering_graph(hyper_data=hyper_data, downscaling_method='UMAP', dim_clusterization=dim_clusterization,
                                      n_clusters=n_clusters, init_params='random', color=color,
                                      download_path=folder_path + '/clusterization' + '/clusterization_umap_random.html',
                                      download_path_table=folder_path + '/clusterization' + '/clusterization_umap_random.xlsx',
                                      with_png=with_png, png_scale=png_scale, n_neighbors_umap=n_neighbors_umap,
                                      png_width=png_width, png_height=png_height,
                                      fig_show=fig_show)
    get_em_algorithm_clustering_graph(hyper_data=hyper_data, downscaling_method='UMAP', dim_clusterization=dim_clusterization,
                                      n_clusters=n_clusters, init_params='kmeans', color=color,
                                      download_path=folder_path + '/clusterization' + '/clusterization_umap_kmeans.html',
                                      download_path_table=folder_path + '/clusterization' + '/clusterization_umap_kmeans.xlsx',
                                      with_png=with_png, png_scale=png_scale, n_neighbors_umap=n_neighbors_umap,
                                      png_width=png_width, png_height=png_height,
                                      fig_show=fig_show)
    
    os.mkdir(folder_path + '/' + 'statistical_analysis')
    get_stat_tests_graph(hyper_data=hyper_data, alpha=alpha, stat_test_names=stat_test_names, factors=factors_pairwise_tests,
                   alternative=alternative, corrected_p_value_method=corrected_p_value_method,
                   params_scipy=params_scipy,
                   color=color,
                   download_dir_path=folder_path + '/statistical_analysis',
                   with_png=with_png, png_scale=png_scale,
                   png_width=png_width, png_height=png_height,
                   fig_show=fig_show)
