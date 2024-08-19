from hyper_img.hyperImg import HyperImg
from hyper_img.hyperImg_functions.get_data_funcs import get_mean_diff_and_confident_interval_df, \
                                                        get_df_graphics_medians_wavelenght, \
                                                        get_df_pca_and_explained_variance, get_df_medians, \
                                                        get_ttest_p_value_df, get_df_isomap, get_df_umap, \
                                                        get_df_em_algorithm_clustering, get_mannwhitneyu_p_value_df, \
                                                        get_chi2_p_value_df, get_boxplot_values

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


def get_boxplot_graphs(hyper_imges: tp.Sequence[HyperImg],
                       channels: tp.Sequence[int] | None = None,
                       color: dict[str, str] | None = None,
                       download_path_dir: str = '',
                       with_png: bool = False,
                       png_scale: float = 8,
                       png_width: int = 500,
                       png_height: int = 700,
                       mean_aggregate: bool = False,
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
    values = get_boxplot_values(hyper_imges, channels, mean_aggregate)

    all_channels = channels if channels is not None else range(len(hyper_imges[0].medians))

    target_variable_names = set()
    for j, img in enumerate(hyper_imges):
        target_variable_names.add(img.target_variable)

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
                                f'boxplot_wavelength:{ch * hyper_imges[0].camera_sensitive + hyper_imges[0].camera_begin_wavelenght}.html')
            fig.write_html(path)
            if with_png:
                change_png_fig(fig)
                pio.write_image(fig, path.replace('.html', '.png'), scale=png_scale,
                                width=png_width, height=png_height)


def get_mean_diff_and_confident_interval_graph(hyper_imges: tp.Sequence[HyperImg],
                                               target_variable_1: str,
                                               target_variable_2: str,
                                               level: float = 0.95,
                                               download_path: str = '',
                                               with_png: bool = False,
                                               png_scale: float = 8,
                                               png_width: int = 500,
                                               png_height: int = 700,
                                               mean_aggregate: bool = False,
                                               fig_show: bool = False) -> None:
    """
    Plots confidence interval for sample mean differences.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        target_variable_1 (str): first target vaiable name.
        target_variable_2 (str): second target vaiable name.
        level (float): level for confident interval. Default 0.95 (95%).
        download_path (str): if not equal to '', then save the graph as download_path.
        with_png (bool): if true, then save plots in png. Default False.
        png_scale (float): text scale in png plot. Default 5.
        png_width (int): width png plot. Default 950.
        png_height (int): height png plot. Default 700.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
        fig_show (bool): is it necessary to show graphs. Defaults to False.
        
    """

    warnings.filterwarnings('ignore')

    m, l, r = get_mean_diff_and_confident_interval_df(hyper_imges, target_variable_1, target_variable_2, level, mean_aggregate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=l['Wavelength'], y=l['Median'],
                             name='left border', line={'color': 'red'}))
    fig.add_trace(go.Scatter(x=r['Wavelength'], y=r['Median'], fill='tonexty',
                             name='right border', line={'color': 'red'}))
    fig.add_trace(go.Scatter(x=m['Wavelength'], y=m['Median'], name='difference between sample means',
                             line={'color': 'blue'}))

    fig.update_layout(title=f'{level * 100}% confident interval for difference between sample means '
                            f'({target_variable_1}, {target_variable_2})',
                      xaxis_title='Wavelength, nm',
                      yaxis_title='Median')
    if fig_show:
        fig.show()

    if download_path:
        fig.write_html(download_path)
        if with_png:
            change_png_fig(fig)
            pio.write_image(fig, download_path.replace('.html', '.png'), scale=png_scale,
                            width=png_width, height=png_height)


def get_medians_wavelenght_graph(hyper_imges: tp.Sequence[HyperImg],
                                 color: dict[str, str] | None = None,
                                 download_path: str = '',
                                 with_png: bool = False,
                                 png_scale: float = 8,
                                 png_width: int = 500,
                                 png_height: int = 700,
                                 mean_aggregate: bool = False,
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

    df = get_df_graphics_medians_wavelenght(hyper_imges, mean_aggregate)

    if color is not None:
        fig = px.line(df, x='Wavelength', y='Median', color=hyper_imges[0].target_varible_name,
                      color_discrete_map=color, line_group=df['Sample'], hover_name='Object name',
                      **kwargs)
    else:
        fig = px.line(df, x='Wavelength', y='Median', color=hyper_imges[0].target_varible_name,
                      line_group=df['Sample'], **kwargs)

    fig.update_layout(title=f'Dependence of medina on wavelength for {hyper_imges[0].target_varible_name}',
                      xaxis_title='Wavelength, nm',
                      yaxis_title='Median')
    fig.update_traces(line=dict(width=0.5))

    if fig_show:
        fig.show()

    if download_path:
        fig.write_html(download_path)
        if with_png:
            change_png_fig(fig)
            pio.write_image(fig, download_path.replace('.html', '.png'), scale=png_scale,
                            width=png_width, height=png_height)


def get_2_pca_graph(hyper_imges: tp.Sequence[HyperImg],
                    color: dict[str, str] | None = None,
                    download_path: str = '',
                    with_png: bool = False,
                    png_scale: float = 8,
                    png_width: int = 500,
                    png_height: int = 700,
                    mean_aggregate: bool = False,
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

    df, var = get_df_pca_and_explained_variance(hyper_imges, 2, mean_aggregate)

    traces = []
    color_dct = dict()
    if color is None:
        for i, name in enumerate(df[hyper_imges[0].target_varible_name].unique()):
            color_dct[name] = plotly_color[i]
    else:
        color_dct = color
    df['group_name'] = df[hyper_imges[0].target_varible_name] + '_' + df['Object name']
    uniq_names = set()
    for group, data in df.groupby('group_name'):
        trace = go.Scatter(
            x=data['1'],
            y=data['2'],
            mode='lines+markers',
            line=dict(color=color_dct[data[hyper_imges[0].target_varible_name].iloc[0]],
                      width=0.3),
            name=data[hyper_imges[0].target_varible_name].iloc[0],
            marker=dict(symbol='circle'),
            text=data['Object name'].iloc[0],
            showlegend=False if data[hyper_imges[0].target_varible_name].iloc[0] in uniq_names else True
        )
        uniq_names.add(data[hyper_imges[0].target_varible_name].iloc[0])
        traces.append(trace)

    fig = go.Figure(data=traces)

    # if color is not None:
    #     fig = px.scatter(df, x='1', y='2', color=hyper_imges[0].target_varible_name, hover_name='Object name',
    #                      color_discrete_map=color, line_group='Object name')
    # else:
    #     fig = px.scatter(df, x='1', y='2', color=hyper_imges[0].target_varible_name, hover_name='Object name',
    #                      line_group='Object name')

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


def get_2_isomap_graph(hyper_imges: tp.Sequence[HyperImg],
		               n_neighbors: int = 5,
                       color: dict[str, str] | None = None,
                       download_path: str = '',
                       with_png: bool = False,
                       png_scale: float = 8,
                       png_width: int = 500,
                       png_height: int = 700,
                       mean_aggregate: bool = False,
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

    df = get_df_isomap(hyper_imges, n_neighbors, mean_aggregate=mean_aggregate, **kwargs)

    traces = []
    color_dct = dict()
    if color is None:
        for i, name in enumerate(df[hyper_imges[0].target_varible_name].unique()):
            color_dct[name] = plotly_color[i]
    else:
        color_dct = color
    df['group_name'] = df[hyper_imges[0].target_varible_name] + '_' + df['Object name']
    uniq_names = set()
    for group, data in df.groupby('group_name'):
        trace = go.Scatter(
            x=data['1'],
            y=data['2'],
            mode='lines+markers',
            line=dict(color=color_dct[data[hyper_imges[0].target_varible_name].iloc[0]],
                      width=0.3),
            name=data[hyper_imges[0].target_varible_name].iloc[0],
            marker=dict(symbol='circle'),
            text=data['Object name'].iloc[0],
            showlegend=False if data[hyper_imges[0].target_varible_name].iloc[0] in uniq_names else True
        )
        uniq_names.add(data[hyper_imges[0].target_varible_name].iloc[0])
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

def get_2_umap_graph(hyper_imges: tp.Sequence[HyperImg],
                     n_neighbors: int = 15,
                     color: dict[str, str] | None = None,
                     download_path: str = '',
                     with_png: bool = False,
                     png_scale: float = 8,
                     png_width: int = 500,
                     png_height: int = 700,
                     mean_aggregate: bool = False,
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

    df = get_df_umap(hyper_imges, n_neighbors, mean_aggregate=mean_aggregate, **kwargs)

    traces = []
    color_dct = dict()
    if color is None:
        for i, name in enumerate(df[hyper_imges[0].target_varible_name].unique()):
            color_dct[name] = plotly_color[i]
    else:
        color_dct = color
    df['group_name'] = df[hyper_imges[0].target_varible_name] + '_' + df['Object name']
    uniq_names = set()
    for group, data in df.groupby('group_name'):
        trace = go.Scatter(
            x=data['1'],
            y=data['2'],
            mode='lines+markers',
            line=dict(color=color_dct[data[hyper_imges[0].target_varible_name].iloc[0]],
                      width=0.3),
            name=data[hyper_imges[0].target_varible_name].iloc[0],
            marker=dict(symbol='circle'),
            text=data['Object name'].iloc[0],
            showlegend=False if data[hyper_imges[0].target_varible_name].iloc[0] in uniq_names else True
        )
        uniq_names.add(data[hyper_imges[0].target_varible_name].iloc[0])
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


def get_mannwhitneyu_p_value_graph(hyper_imges: tp.Sequence[HyperImg],
                                  target_variable_1: str,
                                  target_variable_2: str,
                                  alternative: str = 'two-sided',
                                  corrected_p_value_method: str | None = 'holm',
                                  params_scipy: dict[str, tp.Any] | None = None,
                                  download_path: str = '',
                                  with_png: bool = False,
                                  png_scale: float = 8,
                                  png_width: int = 500,
                                  png_height: int = 700,
                                  mean_aggregate: bool = False,
                                  fig_show: bool = False) -> None:
    """
    Plots p-value of Mann-Whitney U rank test versus wavelength.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        target_variable_1 (str): first target vaiable name.
        target_variable_2 (str): second target vaiable name.
        alternative (str): defines the alternative hypothesis ('two-sided', 'less', 'greater'). Default is ‘two-sided’
        corrected_p_value_method (str | None): method from statsmodel multipletests. Default 'holm'.
        params_scipy (dict[str, tp.Any] | None): dict with params for scipy.stats.mannwhitneyu.
                                                Default None (no extra params).
        download_path (str): if not equal to '', then save the graph as download_path.
        with_png (bool): if true, then save plots in png. Default False.
        png_scale (float): text scale in png plot. Default 5.
        png_width (int): width png plot. Default 950.
        png_height (int): height png plot. Default 700.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
        fig_show (bool): is it necessary to show graphs. Defaults to False.
        """

    df = get_mannwhitneyu_p_value_df(hyper_imges, target_variable_1, target_variable_2, 
                                     alternative, corrected_p_value_method, params_scipy, mean_aggregate)

    fig = px.line(df, x='wavelength, nm', y='p-value',
                  title=f'Mann–Whitney U test ({target_variable_1}, {target_variable_2}), alternative: {alternative}'
                  + f', correction for multiple testing: {corrected_p_value_method}',
                  markers=True)
    if fig_show:
        fig.show()

    if download_path:
        fig.write_html(download_path)
        if with_png:
            change_png_fig(fig)
            pio.write_image(fig, download_path.replace('.html', '.png'), scale=png_scale,
                            width=png_width, height=png_height)


def get_ttest_p_value_graph(hyper_imges: tp.Sequence[HyperImg],
                            target_variable_1: str,
                            target_variable_2: str,
                            alternative: str = 'two-sided',
                            corrected_p_value_method: str | None = 'holm',
                            params_scipy: dict[str, tp.Any] | None = None,
                            download_path: str = '',
                            with_png: bool = False,
                            png_scale: float = 8,
                            png_width: int = 500,
                            png_height: int = 700,
                            mean_aggregate: bool = False,
                            fig_show: bool = False) -> None:
    """
    Plots p-value of independent t-test criterion versus wavelength.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        target_variable_1 (str): first target vaiable name.
        target_variable_2 (str): second target vaiable name.
        alternative (str): defines the alternative hypothesis ('two-sided', 'less', 'greater'). Default is ‘two-sided’
        corrected_p_value_method (str | None): method from statsmodel multipletests. Default 'holm'.
        params_scipy (dict[str, tp.Any] | None): dict with params for scipy.stats.ttest_ind.
                                                Default None (no extra params).
        download_path (str): if not equal to '', then save the graph as download_path.
        with_png (bool): if true, then save plots in png. Default False.
        png_scale (float): text scale in png plot. Default 5.
        png_width (int): width png plot. Default 950.
        png_height (int): height png plot. Default 700.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
        fig_show (bool): is it necessary to show graphs. Defaults to False.
        """

    df = get_ttest_p_value_df(hyper_imges, target_variable_1, target_variable_2, 
                              alternative, corrected_p_value_method, params_scipy, mean_aggregate)

    fig = px.line(df, x='wavelength, nm', y='p-value',
                  title=f't-test criterion ({target_variable_1}, {target_variable_2}), alternative: {alternative}'
                  + f', correction for multiple testing: {corrected_p_value_method}',
                  markers=True)
    if fig_show:
        fig.show()

    if download_path:
        fig.write_html(download_path)
        if with_png:
            change_png_fig(fig)
            pio.write_image(fig, download_path.replace('.html', '.png'), scale=png_scale,
                            width=png_width, height=png_height)


def get_chi2_p_value_graph(hyper_imges: tp.Sequence[HyperImg],
                          target_variable_1: str,
                          target_variable_2: str,
                          corrected_p_value_method: str | None = 'holm',
                          number_bins: int = 5,
                          download_path: str = '',
                          with_png: bool = False,
                          png_scale: float = 8,
                          png_width: int = 500,
                          png_height: int = 700,
                          mean_aggregate: bool = False,
                          fig_show: bool = False) -> None:
    """
    Plots p-value of chi 2 criterion versus wavelength.

    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        target_variable_1 (str): first target vaiable name.
        target_variable_2 (str): second target vaiable name.
        corrected_p_value_method (str | None): method from statsmodel multipletests. Default 'holm'.
        number_bins (int): number of bins. Default 5.
        download_path (str): if not equal to '', then save the graph as download_path.
        with_png (bool): if true, then save plots in png. Default False.
        png_scale (float): text scale in png plot. Default 5.
        png_width (int): width png plot. Default 950.
        png_height (int): height png plot. Default 700.
        mean_aggregate (bool): whether to average samples by object_name. Defaults to False.
        fig_show (bool): is it necessary to show graphs. Defaults to False.
        """

    df = get_chi2_p_value_df(hyper_imges, target_variable_1, target_variable_2, 
                             corrected_p_value_method, number_bins, mean)

    fig = px.line(df, x='wavelength, nm', y='p-value',
                  title=f'Chi-square criterion ({target_variable_1}, {target_variable_2}), number of bins: {number_bins}'
                  + f', correction for multiple testing: {corrected_p_value_method}',
                  markers=True)
    if fig_show:
        fig.show()

    if download_path:
        fig.write_html(download_path)
        if with_png:
            change_png_fig(fig)
            pio.write_image(fig, download_path.replace('.html', '.png'), scale=png_scale,
                            width=png_width, height=png_height)


def get_em_algorithm_clustering_graph(hyper_imges: tp.Sequence[HyperImg],
                                      downscaling_method: str = 'PCA',
                                      dim_clusterization: int = 15,
                                      n_clusters: int | None = None,
                                      n_neighbors_isomap: int = 5,
                                      n_neighbors_umap: int = 15,
                                      filter: tp.Callable = lambda x: True,
                                      n_init: int = 30,
                                      init_params: str = 'random',
                                      color: dict[str, str] | None = None,
                                      download_path: str = '',
                                      download_path_table: str = '',
                                      with_png: bool = False,
                                      png_scale: float = 8,
                                      png_width: int = 500,
                                      png_height: int = 700,
                                      mean_aggregate: bool = False,
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
    df, annotation = get_df_em_algorithm_clustering(hyper_imges, filter, downscaling_method, dim_clusterization,
                                                    n_init, n_clusters, init_params, n_neighbors_isomap,
                                                    n_neighbors_umap, mean_aggregate,
                                                    **kwargs_sklearn_gaussian_mixture)
    if downscaling_method == 'PCA':
        df_2d, _ = get_df_pca_and_explained_variance(hyper_imges)
    elif downscaling_method == 'ISOMAP':
        df_2d = get_df_isomap(hyper_imges, n_neighbors=n_neighbors_isomap)
    elif downscaling_method == 'UMAP':
        df_2d = get_df_umap(hyper_imges, n_neighbors=n_neighbors_umap)
    else:
        raise ThisMethodDoesNotExist('Please choose "PCA", "UMAP" or "ISOMAP" ')

    df_2d[hyper_imges[0].target_varible_name] = pd.Series([tg_v + ' cluster: ' + str(cluster)
                                                           for tg_v, cluster in zip(df[hyper_imges[0
                                                                                         ].target_varible_name].values,
                                                                                    df['Clusters'])])

    if color is not None:
        fig = px.scatter(df_2d, x='1', y='2', color=hyper_imges[0].target_varible_name, hover_name='Object name',
                         color_discrete_map=color)
    else:
        fig = px.scatter(df_2d, x='1', y='2', color=hyper_imges[0].target_varible_name, hover_name='Object name')

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


def create_folder_with_all_graphs(hyper_imges: tp.Sequence[HyperImg],
                                  folder_path: str = 'all_hyper_graphics',
                                  color: dict[str, str] | None = None,
                                  dim_clusterization: int = 10,
                                  n_clusters: int | None = None,
                                  number_bins_chi2: int = 7,
                                  n_neighbors_isomap: int = 5,
                                  n_neighbors_umap: int = 15,
                                  confident_interval_level: float = 0.95,
                                  mannwhitneyu_alternative: str = 'two-sided',
                                  mannwhitneyu_scipy_params: dict[str, tp.Any] | None = None,
                                  ttest_scipy_params: dict[str, tp.Any] | None = None,
                                  corrected_p_value_method: str | None = 'holm',
                                  boxplots_channels: tp.Sequence[int] | None = None,
                                  with_png: bool = False,
                                  png_scale: float = 8,
                                  png_width: int = 500,
                                  png_height: int = 700,
                                  mean_aggregate_dim_reduction: bool = False,
                                  mean_aggregate_visualization: bool = False,
                                  mean_aggregate_stat_anal: bool = True,
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
    get_boxplot_graphs(hyper_imges=hyper_imges, color=color,
                           channels=boxplots_channels,
                           download_path_dir=os.path.join(folder_path, 'boxplots'),
                           with_png=with_png,
                           png_scale=png_scale,
                           png_width=png_width,
                           png_height=png_height,
                           mean_aggregate=mean_aggregate_visualization,
                           fig_show=fig_show)
    get_medians_wavelenght_graph(hyper_imges=hyper_imges, color=color,
                                 download_path=folder_path + '/medians_wavelen.html',
                                 with_png=with_png, png_scale=png_scale,
                                 png_width=png_width, png_height=png_height,
                                 mean_aggregate=mean_aggregate_visualization,
                                 fig_show=fig_show)
    os.mkdir(folder_path + '/visualization')
    get_2_pca_graph(hyper_imges=hyper_imges, color=color, download_path=folder_path + '/visualization' + '/pca.html',
                    with_png=with_png, png_scale=png_scale,
                    png_width=png_width, png_height=png_height,
                    mean_aggregate=mean_aggregate_dim_reduction,
                    fig_show=fig_show)
    get_2_umap_graph(hyper_imges=hyper_imges, color=color, n_neighbors=n_neighbors_umap,
                     download_path=folder_path + '/visualization' + '/umap.html',
                     with_png=with_png, png_scale=png_scale,
                     png_width=png_width, png_height=png_height,
                     mean_aggregate=mean_aggregate_dim_reduction,
                     fig_show=fig_show)
    get_2_isomap_graph(hyper_imges=hyper_imges, n_neighbors=n_neighbors_isomap,
                       color=color, download_path=folder_path + '/visualization' + '/isomap.html',
                       with_png=with_png, png_scale=png_scale,
                       png_width=png_width, png_height=png_height,
                       mean_aggregate=mean_aggregate_dim_reduction,
                       fig_show=fig_show)
    os.mkdir(folder_path + '/clusterization')
    get_em_algorithm_clustering_graph(hyper_imges, downscaling_method='PCA', dim_clusterization=dim_clusterization,
                                      n_clusters=n_clusters, init_params='random', color=color,
                                      download_path=folder_path + '/clusterization' + '/clusterization_pca_random.html',
                                      download_path_table=folder_path + '/clusterization' + '/clusterization_pca_random.xlsx',
                                      with_png=with_png, png_scale=png_scale,
                                      png_width=png_width, png_height=png_height,
                                      mean_aggregate=mean_aggregate_dim_reduction,
                                      fig_show=fig_show)
    get_em_algorithm_clustering_graph(hyper_imges, downscaling_method='PCA', dim_clusterization=dim_clusterization,
                                      n_clusters=n_clusters, init_params='kmeans', color=color,
                                      download_path=folder_path + '/clusterization' + '/clusterization_pca_kmeans.html',
                                      download_path_table=folder_path + '/clusterization' + '/clusterization_pca_kmeans.xlsx',
                                      with_png=with_png, png_scale=png_scale,
                                      png_width=png_width, png_height=png_height,
                                      mean_aggregate=mean_aggregate_dim_reduction,
                                      fig_show=fig_show)
    get_em_algorithm_clustering_graph(hyper_imges, downscaling_method='ISOMAP', dim_clusterization=dim_clusterization,
                                      n_clusters=n_clusters, init_params='random', color=color,
                                      download_path=folder_path + '/clusterization' + '/clusterization_isomap_random.html',
                                      download_path_table=folder_path + '/clusterization' + '/clusterization_isomap_random.xlsx',
                                      with_png=with_png, png_scale=png_scale, n_neighbors_isomap=n_neighbors_isomap,
                                      png_width=png_width, png_height=png_height,
                                      mean_aggregate=mean_aggregate_dim_reduction,
                                      fig_show=fig_show)
    get_em_algorithm_clustering_graph(hyper_imges, downscaling_method='ISOMAP', dim_clusterization=dim_clusterization,
                                      n_clusters=n_clusters, init_params='kmeans', color=color,
                                      download_path=folder_path + '/clusterization' + '/clusterization_isomap_kmeans.html',
                                      download_path_table=folder_path + '/clusterization' + '/clusterization_isomap_kmeans.xlsx',
                                      with_png=with_png, png_scale=png_scale, n_neighbors_isomap=n_neighbors_isomap,
                                      png_width=png_width, png_height=png_height,
                                      mean_aggregate=mean_aggregate_dim_reduction,
                                      fig_show=fig_show)
    get_em_algorithm_clustering_graph(hyper_imges, downscaling_method='UMAP', dim_clusterization=dim_clusterization,
                                      n_clusters=n_clusters, init_params='random', color=color,
                                      download_path=folder_path + '/clusterization' + '/clusterization_umap_random.html',
                                      download_path_table=folder_path + '/clusterization' + '/clusterization_umap_random.xlsx',
                                      with_png=with_png, png_scale=png_scale, n_neighbors_umap=n_neighbors_umap,
                                      png_width=png_width, png_height=png_height,
                                      mean_aggregate=mean_aggregate_dim_reduction,
                                      fig_show=fig_show)
    get_em_algorithm_clustering_graph(hyper_imges, downscaling_method='UMAP', dim_clusterization=dim_clusterization,
                                      n_clusters=n_clusters, init_params='kmeans', color=color,
                                      download_path=folder_path + '/clusterization' + '/clusterization_umap_kmeans.html',
                                      download_path_table=folder_path + '/clusterization' + '/clusterization_umap_kmeans.xlsx',
                                      with_png=with_png, png_scale=png_scale, n_neighbors_umap=n_neighbors_umap,
                                      png_width=png_width, png_height=png_height,
                                      mean_aggregate=mean_aggregate_dim_reduction,
                                      fig_show=fig_show)
    names = np.unique([hyper_imges[i].target_variable for i in range(len(hyper_imges))
                       if hyper_imges[i].target_variable != ''])
    # os.mkdir(folder_path + '/' + 'chi_2')
    os.mkdir(folder_path + '/' + 'ttest')
    os.mkdir(folder_path + '/' + f'{confident_interval_level}_mean_confident_intervals')
    os.mkdir(folder_path + '/' + 'mannwhitneyu')
    lst = []
    for tg_v_1 in names:
        for tg_v_2 in names:
            if tg_v_2 == tg_v_1:
                continue
            if (tg_v_1, tg_v_2) in lst or (tg_v_2, tg_v_1) in lst:
                continue
            # with warnings.catch_warnings():
            #     warnings.filterwarnings('error')
            #     if (tg_v_1, tg_v_2) in lst or (tg_v_2, tg_v_1) in lst:
            #         continue
            #     bins = number_bins_chi2
            #     while True:
            #         try:
            #             get_chi2_p_value_graph(hyper_imges, tg_v_1, tg_v_2, corrected_p_value_method, number_bins=bins,
            #                                     download_path=folder_path + '/' + 'chi_2/' + f'chi_2_{tg_v_1}_{tg_v_2}.html',
            #                                     with_png=with_png, png_scale=png_scale,
            #                                     png_width=png_width, png_height=png_height)
            #         except RuntimeWarning:
            #             bins -= 1
            #         else:
            #             break
            get_mean_diff_and_confident_interval_graph(hyper_imges, tg_v_1, tg_v_2, confident_interval_level,
                                                       download_path=folder_path + '/' +
                                                                     f'{confident_interval_level}_mean_confident_intervals'
                                                                     + f'/conf_int_{tg_v_1}_{tg_v_2}.html',
                                                       with_png=with_png, png_scale=png_scale,
                                                       png_width=png_width, png_height=png_height,
                                                       mean_aggregate=mean_aggregate_stat_anal,
                                                       fig_show=fig_show)
            get_mannwhitneyu_p_value_graph(hyper_imges, tg_v_1, tg_v_2, mannwhitneyu_alternative,
                                           corrected_p_value_method,
                                           mannwhitneyu_scipy_params,
                                           download_path=folder_path + '/' + 'mannwhitneyu' +
                                                         f'/mannwhitneyu_{tg_v_1}_{tg_v_2}.html',
                                           with_png=with_png, png_scale=png_scale,
                                           png_width=png_width, png_height=png_height,
                                           mean_aggregate=mean_aggregate_stat_anal,
                                           fig_show=fig_show)
            get_ttest_p_value_graph(hyper_imges, tg_v_1, tg_v_2, mannwhitneyu_alternative,
                                    corrected_p_value_method,
                                    ttest_scipy_params,
                                    download_path=folder_path + '/' + 'ttest' +
                                                  f'/ttest_{tg_v_1}_{tg_v_2}.html',
                                    with_png=with_png, png_scale=png_scale,
                                    png_width=png_width, png_height=png_height,
                                    mean_aggregate=mean_aggregate_stat_anal,
                                    fig_show=fig_show)
            lst.append((tg_v_1, tg_v_2))
