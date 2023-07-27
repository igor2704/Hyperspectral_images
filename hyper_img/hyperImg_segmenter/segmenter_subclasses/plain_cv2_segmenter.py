from hyper_img.hyperImg_segmenter.segmenter import Segmenter
from hyper_img.hyperImg import HyperImg

import cv2
import numpy as np


class PlainCv2Segmenter(Segmenter):

    def __init__(self, threshold: float = 25):
        """
        Args:
            threshold (float): cv2 threshold.
        """
        self.threshold = threshold

    def get_mask(self, hyper_image: np.ndarray) -> np.ndarray:
        bgr = HyperImg.get_bgr(hyper_image)
        img_black = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, im_thr = cv2.threshold(img_black, self.threshold, 255, cv2.THRESH_BINARY)
        return im_thr.clip(0, 1)
