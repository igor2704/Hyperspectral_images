from hyper_img.hyperImg import HyperImg
from hyper_img.hyperImg_functions.get_data_funcs import get_df_medians, get_df_pca_and_explained_variance, \
                                                        get_df_umap, get_df_isomap

import typing as tp

import numpy as np
import pandas as pd

import sklearn

from sklearn.model_selection import train_test_split, GridSearchCV

from catboost import CatBoostClassifier

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.manifold import Isomap

import umap

import matplotlib.pyplot as plt
import seaborn as sns

import os


class EmptyHyperImgList(Exception):
    pass


class NeedHyperImgSubclass(Exception):
    pass


class ThisMethodDoesNotExist(Exception):
    pass


class NeedTest(Exception):
    pass


def get_average_metrics(models: tp.Sequence[sklearn.base.BaseEstimator],
                        models_name: tp.Sequence[str],
                        X: np.ndarray,
                        y: np.ndarray,
                        class_function: tp.Callable[[str], tp.Sequence[int]],
                        ind_to_text: dict[int, str]) -> pd.DataFrame:
    """
    Get classification results by average metrics using the function converting target variable to integer sequence.
    Args:
        models (tp.Sequence[sklearn.base.BaseEstimator]): classification models
        models_name (tp.Sequence[str]): models names
        X (np.ndarray): features
        y (np.ndarray): classes
        class_function (tp.Callable[[str], tp.Sequence[int]]): function converting target variable to integer sequence
        ind_to_text (dict[int, str]): dictionary for translating the numerical representation
                                      of a feature into a textual one

    Returns:
        pd.DataFrame: table with classification results
    """

    y_classes = np.array([class_function(y_i) for y_i in y])
    y_predicts = []
    for model in models:
        try:
            y_predicts.append([ind_to_text[x] for x in model.predict(X)])
        except TypeError:
            y_predicts.append([ind_to_text[x[0]] for x in model.predict(X)])
    y_predicts_classes = []
    for Y_i in y_predicts:
        y_predicts_classes.append([class_function(y_i) for y_i in Y_i])
    y_predicts_classes = np.array(y_predicts_classes)
    try:
        num_class = len(class_function(y[0]))
    except KeyError:
        num_class = len(class_function(y.iloc[0]))
    accuracy = [0] * len(models)
    f1 = [0] * len(models)
    precision = [0] * len(models)
    recall = [0] * len(models)
    for cl in range(num_class):
        for i, y_pred in enumerate(y_predicts_classes):
            accuracy[i] += accuracy_score(y_classes[:, cl], 
                                          y_pred[:, cl]) / num_class
            f1[i] += f1_score(y_classes[:, cl], 
                              y_pred[:, cl]) / num_class
            precision[i] += precision_score(y_classes[:, cl], 
                                            y_pred[:, cl]) / num_class
            recall[i] += recall_score(y_classes[:, cl], 
                                      y_pred[:, cl]) / num_class
    return pd.DataFrame(zip(models_name, accuracy, f1, precision, recall),
                        columns=['model', 'accuracy', 'f1', 'precision', 'recall'])  

    
def get_macro_metrics(models: tp.Sequence[sklearn.base.BaseEstimator],
                      models_name: tp.Sequence[str],
                      X: np.ndarray,
                      y: np.ndarray) -> pd.DataFrame:
    """
    Get classification results by macro metrics.
    Args:
        models (tp.Sequence[sklearn.base.BaseEstimator]): classification models
        models_name (tp.Sequence[str]): models names
        X (np.ndarray): features
        y (np.ndarray): classes

    Returns:
        pd.DataFrame: table with classification results
    """
    y_predicts = [model.predict(X) for model in models]
    accuracy: tp.List[float] = [accuracy_score(y, y_pr) for y_pr in y_predicts]
    f1: tp.List[float] = [f1_score(y, y_pr, average='macro') for y_pr in y_predicts]
    precision: tp.List[float] = [precision_score(y, y_pr, average='macro') for y_pr in y_predicts]
    recall: tp.List[float] = [recall_score(y, y_pr, average='macro') for y_pr in y_predicts]
    return pd.DataFrame(zip(models_name, accuracy, f1, precision, recall),
                        columns=['model', 'accuracy', 'f1', 'precision', 'recall'])


def get_micro_metrics(models: tp.Sequence[sklearn.base.BaseEstimator],
                      models_name: tp.Sequence[str],
                      X: np.ndarray,
                      y: np.ndarray) -> pd.DataFrame:
    """
    Get classification results by micro metrics.
    Args:
       models (tp.Sequence[sklearn.base.BaseEstimator]): classification models
       models_name (tp.Sequence[str]): models names
       X (np.ndarray): features
       y (np.ndarray): classes

    Returns:
       pd.DataFrame: table with classification results
       """
    y_predicts = [model.predict(X) for model in models]
    accuracy: tp.List[float] = [accuracy_score(y, y_pr) for y_pr in y_predicts]
    f1: tp.List[float] = [f1_score(y, y_pr, average='micro') for y_pr in y_predicts]
    precision: tp.List[float] = [precision_score(y, y_pr, average='micro') for y_pr in y_predicts]
    recall: tp.List[float] = [recall_score(y, y_pr, average='micro') for y_pr in y_predicts]
    return pd.DataFrame(zip(models_name, accuracy, f1, precision, recall),
                        columns=['model', 'accuracy', 'f1', 'precision', 'recall'])


def get_binary_metrics(models: tp.Sequence[sklearn.base.BaseEstimator],
                       models_name: tp.Sequence[str],
                       X: np.ndarray,
                       y: np.ndarray) -> pd.DataFrame:
    """
    Get binary classification results.
    Args:
       models (tp.Sequence[sklearn.base.BaseEstimator]): classification models
       models_name (tp.Sequence[str]): models names
       X (np.ndarray): features
       y (np.ndarray): classes

    Returns:
       pd.DataFrame: table with classification results
       """
    y_predicts = [model.predict(X) for model in models]
    accuracy: tp.List[float] = [accuracy_score(y, y_pr) for y_pr in y_predicts]
    f1: tp.List[float] = [f1_score(y, y_pr, average='binary') for y_pr in y_predicts]
    precision: tp.List[float] = [precision_score(y, y_pr, average='binary') for y_pr in y_predicts]
    recall: tp.List[float] = [recall_score(y, y_pr, average='binary') for y_pr in y_predicts]
    return pd.DataFrame(zip(models_name, accuracy, f1, precision, recall),
                        columns=['model', 'accuracy', 'f1', 'precision', 'recall'])


def get_table_res_and_confusion_matrix(hyper_imges: tp.Sequence[HyperImg],
                                       test_size: float | None = 0.15,
                                       test_hyper_images: tp.Sequence[HyperImg] | None = None,
                                       train_X: pd.DataFrame | None = None,
                                       test_X: pd.DataFrame | None = None,
                                       train_y: pd.DataFrame | None = None,
                                       test_y: pd.DataFrame | None = None,
                                       shuffle_test: bool = True,
                                       cv: int = 4,
                                       filter: tp.Callable = lambda x: True,
                                       downscaling_method: str | None = None,
                                       n_components: int | None = None,
                                       parameters_random_forest: dict[str, list[float]] | None = None,
                                       parameters_catboost: dict[str, list[float]] | None = None,
                                       alpha_ridge_regression: list[float] | None = None,
                                       early_stopping_rounds_catboost: int = 10,
                                       class_function: tp.Callable[[str], tp.Sequence] | None = None,
                                       kwargs_downscaling: dict[str, tp.Any] | None = None,
                                       n_neighbors_isomap: int = 5,
                                       n_neighbors_umap: int = 15,
                                       save_path_folder: str = '') -> tuple[dict[str, pd.DataFrame],
                                                                            dict[str, pd.DataFrame]]:
    """
    Get classification results by main metrics and confusion matrix.
    Using models such as logistic regression, random forest, catboost.
    Args:
        hyper_imges (tp.Sequence[HyperImg]): the sequence of hyperspectral images.
        test_size (float): test sample percentage. Default 0.15.
        shuffle_test (bool): shuffle test. Default True.
        cv (int): number of cross validation folds. Default 4.
        filter (tp.Callable): filter function.
        downscaling_method (str | None ): downscaling method (None (no downscaling), 'PCA', 'UMAP', 'ISOMAP').
                                          Default None.
        n_components (int | None): space dimension after downscaling.
        parameters_random_forest (dict[str, list[float]] | None): parameters for random forest in sklearn.
                                                                  Default None (no parameters).
        parameters_catboost: parameters for catboost in catboost.
                             Default None (no parameters).
        alpha_ridge_regression (list[float] | None): Values fo constant that multiplies the L2 term,
                                                     controlling regularization strength in ridge regression.
        early_stopping_rounds_catboost (int): early stopping rounds catboost (to prevent overfitting). Default 10.
        class_function (tp.Callable[[str], tp.Sequence] | None): function converting target variable to integer
                                                                 sequence. Default None (don`t show average metrics).
        kwargs_downscaling (dict[str, tp.Any] | None): parameters for sklearn isomap, pca or for umap. Defaults None.
        n_neighbors_isomap (int): number of neighbors to consider for each point, if using isomap. Default 5.
        n_neighbors_umap (int): number of neighbors to consider for each point, if using umap. Default 15.
        save_path_folder (str): If save_path != '', then create folder and save all tables and confusion matrix
                                in save_path_folder. Default ''.

    Returns:
        dict[str, pd.DataFrame]: tables with results.
        dict[str, pd.DataFrame]: confusion matrixes.
    """
    if len(hyper_imges) == 0:
        raise EmptyHyperImgList

    if not issubclass(type(hyper_imges[0]), HyperImg):
        raise NeedHyperImgSubclass

    if test_hyper_images is None and test_size is None:
        raise NeedTest('test_hyper_images or test_size need to be is not None')

    if test_hyper_images is not None and test_size is not None:
        raise NeedTest('test_hyper_images or test_size need to be is None')

    if kwargs_downscaling is None:
        kwargs_downscaling = dict()

    if parameters_random_forest is None:
        parameters_random_forest = {'max_depth': [3, 6],
                                    'min_samples_split': [1, 5],
                                    'min_samples_leaf': [1, 5],
                                    'n_estimators': [150]}
    if parameters_catboost is None:
        parameters_catboost = {'learning_rate': np.arange(0.33, 1, 0.2),
                               'iterations': [150]}
    if alpha_ridge_regression is None:
        alpha_ridge_regression = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

    filtred_hyper_imges = [img for img in hyper_imges if filter(img)]
    if test_hyper_images is not None:
        filtred_hyper_imges_test = [img for img in test_hyper_images if filter(img)]

    if downscaling_method is not None and n_components is not None:
        if downscaling_method == 'PCA':
            pipe = Pipeline([('scaler', StandardScaler(with_std=False)),
                             ('pca', PCA(n_components=n_components, **kwargs_downscaling))])
        elif downscaling_method == 'ISOMAP':
            pipe = Pipeline([('scaler', StandardScaler(with_std=False)),
                             ('isomap', Isomap(n_components=n_components, n_neighbors=n_neighbors_isomap,
                                               **kwargs_downscaling))])
        elif downscaling_method == 'UMAP':
            pipe = Pipeline([('scaler', StandardScaler(with_std=False)),
                             ('umap', umap.UMAP(n_components=n_components, n_neighbors=n_neighbors_umap,
                                                **kwargs_downscaling))])
        else:
            raise ThisMethodDoesNotExist('Please choose "PCA", "ISOMAP" or "UMAP"')
    else:
        pipe = Pipeline([('scaler', StandardScaler(with_std=False))])

    idx_to_target: dict[int, str] = dict()
    target_to_idx: dict[str, int] = dict()

    if test_X is None or test_y is None or train_X is None or train_y is None:
        df = get_df_medians(filtred_hyper_imges)
        if test_hyper_images is not None:
            df_test = get_df_medians(filtred_hyper_imges_test)

        if test_size is not None:
            X = df.drop([hyper_imges[0].target_varible_name, 'Object name'], axis=1)
            y = df[hyper_imges[0].target_varible_name]

            for i, target in enumerate(pd.unique(df[hyper_imges[0].target_varible_name])):
                idx_to_target[i] = target
                target_to_idx[target] = i

            y = y.apply(lambda x: target_to_idx[x])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, shuffle=shuffle_test)
        else:
            X_train = df.drop([hyper_imges[0].target_varible_name, 'Object name'], axis=1)
            y_train = df[hyper_imges[0].target_varible_name]

            X_test = df_test.drop([hyper_imges[0].target_varible_name, 'Object name'], axis=1)
            y_test = df_test[hyper_imges[0].target_varible_name]

            for i, target in enumerate(np.unique(np.concatenate([pd.unique(df[hyper_imges[0].target_varible_name]),
                                                                 pd.unique(
                                                                     df_test[hyper_imges[0].target_varible_name])]))):
                idx_to_target[i] = target
                target_to_idx[target] = i

            y_train = y_train.apply(lambda x: target_to_idx[x])
            y_test = y_test.apply(lambda x: target_to_idx[x])
    else:
        X_train, X_test, y_train, y_test = train_X, test_X, train_y, test_y
        for i, target in enumerate(np.unique(np.concatenate([pd.unique(y_test),
                                                             pd.unique(y_train)]))):
            idx_to_target[i] = target
            target_to_idx[target] = i

        y_train = y_train.apply(lambda x: target_to_idx[x])
        y_test = y_test.apply(lambda x: target_to_idx[x])

    X_train = pipe.fit_transform(X_train)
    X_test = pipe.transform(X_test)

    logistic_regression = LogisticRegressionCV(cv=cv)
    ridge_regression = RidgeClassifierCV(alphas=alpha_ridge_regression)

    logistic_regression.fit(X_train, y_train)
    ridge_regression.fit(X_train, y_train)

    random_forest = GridSearchCV(RandomForestClassifier(), parameters_random_forest, cv=cv)
    random_forest.fit(X_train, y_train)

    catboost = GridSearchCV(CatBoostClassifier(silent=True,
                                               early_stopping_rounds=early_stopping_rounds_catboost),
                                               parameters_catboost, cv=cv)
    catboost.fit(X_train, y_train)

    table_dfs: dict[str, pd.DataFrame] = dict()
    params = ([logistic_regression, ridge_regression,
               random_forest.best_estimator_, catboost.best_estimator_],
              ['Logistic regression', 'Ridge regression',
               'Random forest', 'Catboost'])
    if len(idx_to_target) > 2:
        table_dfs['macro_train'] = get_macro_metrics(*params, X_train, y_train.values)
        table_dfs['micro_train'] = get_micro_metrics(*params, X_train, y_train.values)
        table_dfs['macro_test'] = get_macro_metrics(*params, X_test, y_test.values)
        table_dfs['micro_test'] = get_micro_metrics(*params, X_test, y_test.values)
        if class_function is not None:
            y_train_str = y_train.apply(lambda x: idx_to_target[x])
            y_test_str = y_test.apply(lambda x: idx_to_target[x])
            table_dfs['average_train'] = get_average_metrics(*params, X_train,
                                                            y_train_str, class_function, idx_to_target)
            table_dfs['average_test'] = get_average_metrics(*params, X_test,
                                                            y_test_str, class_function, idx_to_target)
    else:
        table_dfs['binary_train'] = get_binary_metrics(*params, X_train, y_train.values)
        table_dfs['binary_test'] = get_binary_metrics(*params, X_test, y_test.values)

    confusion_matrixes: dict[str, np.ndarray] = dict()
    confusion_matrixes['Logistic regression'] = confusion_matrix(y_test, logistic_regression.predict(X_test))
    confusion_matrixes['Ridge regression'] = confusion_matrix(y_test, ridge_regression.predict(X_test))
    confusion_matrixes['Random forest'] = confusion_matrix(y_test, random_forest.predict(X_test))
    confusion_matrixes['Catboost'] = confusion_matrix(y_test, catboost.predict(X_test))

    for k in confusion_matrixes:
        confusion_matrixes[k] = pd.DataFrame(confusion_matrixes[k], columns=list(target_to_idx.keys()))
        confusion_matrixes[k].index = list(target_to_idx.keys())

    if save_path_folder != '':
        os.mkdir(save_path_folder)
        for table in table_dfs:
            table_dfs[table].to_excel(save_path_folder + f'/{table}.xlsx')
        for matrix in confusion_matrixes:
            plt.figure(figsize=(9, 7))
            plot = sns.heatmap(confusion_matrixes[matrix], annot=True, cbar=False)
            plot.figure.savefig(save_path_folder + '/' + matrix.replace(' ', '_').lower() + '_confusion_matrix.png',
                                pad_inches=1)

    return table_dfs, confusion_matrixes
