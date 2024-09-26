from hyper_img.hyperImg_segmenter.segmenter import Segmenter

from abc import ABC, abstractmethod

import typing as tp

from itertools import product
from copy import deepcopy

import cv2
import numpy as np
import tifffile as tiff
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class EmptyHyperImgList(Exception):
    pass


class NeedHyperImgSubclass(Exception):
    pass


class HyperImg(ABC):
    """
    Abstract class for hyperspectral image with basic methods.
    """

    def __init__(self,
                 path: str,
                 segmenter: Segmenter,
                 savgol_par: tuple[int, int] = (9, 3),
                 black_calibration_img_name: str = '',
                 white_calibration_img_name: str = '',
                 camera_sensitive: int = 4,
                 camera_begin_wavelenght: int = 450,
                 factors: dict[str, tp.Any] | None = None,
                 object_name: str | None = None) -> None:
        """
        Args:
            path (str): images path.
            segmenter (Segmenter): object for segmentation.
            savgol_par (tuple[int, int], optional): parametrs for savgol method. Defaults to (9, 3).
            black_calibration_img_name (str): black image for calibration name. Defaults to ''.
            white_calibration_img_name (str): white image for calibration name. Defaults to ''.
            mask (None | np.ndarray): image mask.
            with_median (bool): if true, then calculate medians for each channel. Defaults True.
            object_name (str): object name. Default None
        """
        self.factors = factors
        self.camera_sensitive = camera_sensitive
        self.camera_begin_wavelenght = camera_begin_wavelenght
        self.object_name = object_name
        self.savgol_par = savgol_par
        self.path = path
        self.black_calibration_img_name = black_calibration_img_name
        self.white_calibration_img_name = white_calibration_img_name
        self.img = self._get_image()
        self.widht = self.img.shape[0]
        self.height = self.img.shape[1]
        self.mask = segmenter.get_mask(self.img)

    @staticmethod
    def wave_len(x: int, step: int = 4, begin_wave_len: int = 450) -> int:
        """
        Args:
            x (int): wavelength.
            step (int, optional): step (camera settings). Defaults to 4.
            begin_wave_len (int, optional): initial wavelength (camera settings). Defaults to 450.

        Returns:
            int: channel number.
        """
        return int((x - begin_wave_len) // step)

    @abstractmethod
    def _get_image(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: tiff image.
        """
        pass


    def get_pixels(self) -> tp.List[tp.Tuple[int, int]]:
        """
        Returns:
            tp.List[tp.Tuple[int, int]]: mask pixels.
        """
        return [(x, y) for x, y in product(range(self.widht), range(self.height))
                if self.mask[x, y] != 0]

    def get_medians(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: vector of medians.
        """
        medians: tp.List[np.ndarray] = list()
        pixels = self.get_pixels()
        for i in range(self.img.shape[2]):
            medians.append(np.median(np.array([self.img[p[0]][p[1]][i] for p in pixels])))
        return savgol_filter(np.array(medians), *self.savgol_par)

    def get_all_values(self) -> np.ndarray:
        values: tp.List[np.ndarray] = []
        pixels = self.get_pixels()
        for i in range(self.img.shape[2]):
            values.append(np.array([self.img[p[0]][p[1]][i] for p in pixels]))
        return np.array(values)

    def get_factors(self) -> str:
        factors_name = ''
        for f in self.factors:
            factors_name += f' {f}({self.factors[f]});'
        return factors_name

    @property
    def pan(self) -> np.ndarray:
        """
        Returns:
            _pan.tiff image
        """
        path = self.path.split('.tiff')[0]
        path = path[:-4]
        path += 'pan.tiff'
        return tiff.imread(path)

    @property
    def panimage(self) -> np.ndarray:
        """
        Returns:
            _PANIMAGE.tiff image
        """
        path = self.path.split('.tiff')[0]
        path = path[:-4]
        path += 'PANIMAGE.tiff'
        return tiff.imread(path)

    @staticmethod
    def get_bgr(hyper_image: np.ndarray):
        """
        Args:
            hyper_image (np.ndarray): multichannel hyperspectral image.
        Returns:
            np.ndarray: bgr image representation.
        """
        # To accurately display colors, you need to choose constants

        im_r = hyper_image[:, :, HyperImg.wave_len(630)]
        im_g = hyper_image[:, :, HyperImg.wave_len(510)]
        im_b = hyper_image[:, :, HyperImg.wave_len(450)]

        im_r = (im_r / im_r.max()) * 255
        im_g = (im_g / im_g.max()) * 255
        im_b = (im_b / im_b.max()) * 255

        im_r = np.clip(im_r, 0, 255).astype(np.uint8)
        im_g = np.clip(im_g, 0, 255).astype(np.uint8)
        im_b = np.clip(im_b, 0, 255).astype(np.uint8)

        im_bgr = np.zeros((hyper_image.shape[0], hyper_image.shape[1], 3), dtype=np.uint8)
        im_bgr[:, :, 0] = im_b
        im_bgr[:, :, 1] = im_g
        im_bgr[:, :, 2] = im_r

        return im_bgr

    @property
    def bgr(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: bgr image representation.
        """
        return HyperImg.get_bgr(self.img)

    T = tp.TypeVar('T', 'HyperImg', float, np.ndarray)

    def __add__(self, other: T) -> 'HyperImg':
        new_hyper_img = deepcopy(self)
        if issubclass(type(other), HyperImg):
            new_hyper_img.img += other.img
        else:
            if type(other) == np.ndarray:
                for i in range(len(other)):
                    new_hyper_img.img[:, :, i] += other[i]
            else:
                new_hyper_img.img += other
        if self.with_median:
            new_hyper_img.medians = new_hyper_img.get_medians()
        return new_hyper_img

    def __sub__(self, other: T) -> 'HyperImg':
        new_hyper_img = deepcopy(self)
        if issubclass(type(other), HyperImg):
            new_hyper_img.img -= other.img
        else:
            if type(other) == np.ndarray:
                for i in range(len(other)):
                    new_hyper_img.img[:, :, i] -= other[i]
            else:
                new_hyper_img.img -= other
        if self.with_median:
            new_hyper_img.medians = new_hyper_img.get_medians()
        return new_hyper_img

    def __mul__(self, other: 'HyperImg') -> 'HyperImg':
        new_hyper_img = deepcopy(self)
        if issubclass(type(other), HyperImg):
            new_hyper_img.img *= other.img
        else:
            if type(other) == np.ndarray:
                for i in range(len(other)):
                    new_hyper_img.img[:, :, i] *= other[i]
            else:
                new_hyper_img.img *= other
        if self.with_median:
            new_hyper_img.medians = new_hyper_img.get_medians()
        return new_hyper_img

    def __truediv__(self, other: 'HyperImg') -> 'HyperImg':
        new_hyper_img = deepcopy(self)
        if issubclass(type(other), HyperImg):
            new_hyper_img.img /= other.img
        else:
            if type(other) == np.ndarray:
                for i in range(len(other)):
                    new_hyper_img.img[:, :, i] /= other[i]
            else:
                new_hyper_img.img /= other
        if self.with_median:
            new_hyper_img.medians = new_hyper_img.get_medians()
        return new_hyper_img

    def __str__(self) -> str:
        return f'Hyperspectral image. ' \
               f'\n Image shape: {self.img.shape}.' \
               + self.get_factors()

    def __repr__(self) -> str:
        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB), cmap='gray')
        axes[0].set_title('rgb visualization')

        axes[1].imshow(self.mask, cmap='gray')
        axes[1].set_title('mask segmentation')

        return self.get_factors()
