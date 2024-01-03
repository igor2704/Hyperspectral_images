from hyper_img.hyperImg_segmenter.segmenter import Segmenter

import cv2
import numpy as np


class PlainCv2Segmenter(Segmenter):

    def __init__(self, threshold: float = 25):
        """
        Args:
            threshold (float): cv2 threshold.
        """
        self.threshold = threshold

    @staticmethod
    def _get_bgr(hyper_image: np.ndarray):
        """
        Args:
            hyper_image (np.ndarray): multichannel hyperspectral image.
        Returns:
            np.ndarray: bgr image representation.
        """
        # To accurately display colors, you need to choose constants

        im_r = hyper_image[:, :, PlainCv2Segmenter._wave_len(630)]
        im_g = hyper_image[:, :, PlainCv2Segmenter._wave_len(510)]
        im_b = hyper_image[:, :, PlainCv2Segmenter._wave_len(450)]

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

    @staticmethod
    def _wave_len(x: int, step: int = 4, begin_wave_len: int = 450) -> int:
        """
        Args:
            x (int): wavelength.
            step (int, optional): step (camera settings). Defaults to 4.
            begin_wave_len (int, optional): initial wavelength (camera settings). Defaults to 450.

        Returns:
            int: channel number.
        """
        return int((x - begin_wave_len) // step)

    def get_mask(self, hyper_image: np.ndarray) -> np.ndarray:
        bgr = PlainCv2Segmenter._get_bgr(hyper_image)
        img_black = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, im_thr = cv2.threshold(img_black, self.threshold, 255, cv2.THRESH_BINARY)
        return im_thr.clip(0, 1)
