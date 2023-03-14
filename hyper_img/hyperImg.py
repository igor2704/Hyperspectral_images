from abc import ABC, abstractmethod

import typing as tp

from itertools import product
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class EmptyHyperImgList(Exception):
    pass


class HyperImg(ABC):
    """
    Abstract class for hyperspectral image with basic methods.
    """

    def __init__(self,
                 path: str,
                 threshold_value: float = 25,
                 savgol_par: tuple[int, int] = (9, 3),
                 target_varible_name: str = 'Target Varible',
                 black_calibration_img_name: str = '',
                 white_calibration_img_name: str = '',
                 mask: None | np.ndarray = None) -> None:
        """
        Args:
            path (str): images path.
            threshold_value (float, optional): threshold. Defaults to 25.
            savgol_par (tuple[int, int], optional): parametrs for savgol method. Defaults to (9, 3).
            target_varible_name (str, optional): name of target varible. Defaults to 'Target Varible'.
            black_calibration_img_name (str): black image for calibration name. Defaults to ''.
            white_calibration_img_name (str): white image for calibration name. Defaults to ''.
            mask (None | np.ndarray): image mask.
        """
        self.savgol_par = savgol_par
        self._threshold_value = threshold_value
        self.path = path
        self.target_varible_name = target_varible_name
        self.black_calibration_img_name = black_calibration_img_name
        self.white_calibration_img_name = white_calibration_img_name
        self.img = self._get_tiff()
        self.widht = self.img.shape[0]
        self.height = self.img.shape[1]
        self.pixels = self._get_pixels()
        self.medians = self._get_medians()
        self.target_varible = self._get_target_varible()
        if mask is None:
            self.mask = self._get_mask()
        else:
            self.mask = mask

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
    def _get_tiff(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: tiff image.
        """
        pass

    @abstractmethod
    def _get_target_varible(self) -> str:
        """
        Should be overridden if necessary.

        Returns:
            str: target variable.
        """
        pass

    def _get_mask(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: image mask.
        """
        return self.threshold_bgr.clip(0, 1)

    def _get_pixels(self) -> tp.List[tp.Tuple[int, int]]:
        """
        Returns:
            tp.List[tp.Tuple[int, int]]: mask pixels.
        """
        return [(x, y) for x, y in product(range(self.widht), range(self.height))
                if self.threshold_bgr[x, y] != 0]

    def _get_medians(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: vector of medians.
        """
        medians: tp.List[np.ndarray] = list()
        for i in range(self.img.shape[2]):
            medians.append(np.median(np.array([self.img[p[0]][p[1]][i] for p in self.pixels])))
        return savgol_filter(np.array(medians), *self.savgol_par)

    @property
    def bgr(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: bgr image representation.
        """

        # To accurately display colors, you need to choose constants

        im_r = self.img[:, :, HyperImg.wave_len(630)]
        im_g = self.img[:, :, HyperImg.wave_len(510)]
        im_b = self.img[:, :, HyperImg.wave_len(450)]

        im_r = (im_r / im_r.max()) * 255
        im_g = (im_g / im_g.max()) * 255
        im_b = (im_b / im_b.max()) * 255

        im_r = np.clip(im_r, 0, 255).astype(np.uint8)
        im_g = np.clip(im_g, 0, 255).astype(np.uint8)
        im_b = np.clip(im_b, 0, 255).astype(np.uint8)

        im_bgr = np.zeros((self.widht, self.height, 3), dtype=np.uint8)
        im_bgr[:, :, 0] = im_b
        im_bgr[:, :, 1] = im_g
        im_bgr[:, :, 2] = im_r

        return im_bgr

    @property
    def threshold_bgr(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: mask of image.
        """
        im_black = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        _, im_thr = cv2.threshold(im_black, self._threshold_value, 255, cv2.THRESH_BINARY)
        return im_thr

    def __add__(self, other: 'HyperImg') -> 'HyperImg':
        new_hyper_img = deepcopy(self)
        new_hyper_img.medians += other.medians
        new_hyper_img.pixels.extend(other.pixels)
        new_hyper_img.img += other.img
        new_hyper_img.mask = (self.mask.clip(0, 1) + other.mask.clip(0, 1)).clip(0, 1)
        new_hyper_img._threshold_value = max(new_hyper_img._threshold_value, other._threshold_value)
        new_hyper_img.target_varible = new_hyper_img.target_varible + ' + ' + other.target_varible
        new_hyper_img.path = new_hyper_img.path + ' + ' + other.path
        return new_hyper_img

    def __sub__(self, other: 'HyperImg') -> 'HyperImg':
        new_hyper_img = deepcopy(self)
        new_hyper_img.medians -= other.medians
        new_hyper_img.img -= other.img
        new_hyper_img.img = new_hyper_img.img.clip(a_min=0)
        mask = self.mask.clip(0, 1).astype(int) - other.mask.clip(0, 1).astype(int)
        new_hyper_img.mask = mask.clip(0, 1)
        new_hyper_img.target_varible = new_hyper_img.target_varible + ' - ' + other.target_varible
        new_hyper_img.path = new_hyper_img.path + ' - ' + other.path
        return new_hyper_img

    def __mul__(self, other: 'HyperImg') -> 'HyperImg':
        new_hyper_img = deepcopy(self)
        new_hyper_img.medians *= other.medians
        new_hyper_img.img *= other.img
        new_hyper_img.mask = (self.mask.clip(0, 1) * other.mask.clip(0, 1))
        new_hyper_img.target_varible = new_hyper_img.target_varible + ' * ' + other.target_varible
        new_hyper_img.path = new_hyper_img.path + ' * ' + other.path
        return new_hyper_img

    def __truediv__(self, other: 'HyperImg') -> 'HyperImg':
        new_hyper_img = deepcopy(self)
        try:
            new_hyper_img.medians /= other.medians
        except ZeroDivisionError:
            raise ZeroDivisionError('median division by zero')
        new_hyper_img.img = np.where(other.img != 0, new_hyper_img.img / other.img, 0)
        new_hyper_img.target_varible = new_hyper_img.target_varible + ' / ' + other.target_varible
        new_hyper_img.path = new_hyper_img.path + ' / ' + other.path
        return new_hyper_img

    def __str__(self) -> str:
        return f'Hyperspectral image. ' \
               f'\n {self.target_varible_name} : {self.target_varible} ' \
               f'\n Image shape: {self.img.shape}.'

    def __repr__(self) -> str:
        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(self.bgr, cmap='gray')
        axes[0].set_title('rgb visualization')

        axes[1].imshow(self.mask, cmap='gray')
        axes[1].set_title('mask segmentation')

        return self.target_varible_name + ': ' + self.target_varible
