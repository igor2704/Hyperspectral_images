from hyper_img.hyperImg_segmenter.segmenter import Segmenter

import cv2
import numpy as np


class CircleCv2Segmenter(Segmenter):

    def __init__(self, threshold_circle: float = 30, threshold: float = 90):
        """
        Args:
            threshold_circle (float): cv2 threshold threshold for selecting a circular area.
            threshold (float): threshold for segmentation within a circular region
        """
        self.threshold_circle = threshold_circle
        self.threshold = threshold

    @staticmethod
    def _get_bgr(hyper_image: np.ndarray, step: int = 4, begin_wave_len: int = 450):
        """
        Args:
            hyper_image (np.ndarray): multichannel hyperspectral image.
            step (int, optional): step (camera settings). Defaults to 4.
            begin_wave_len (int, optional): initial wavelength (camera settings). Defaults to 450.
        Returns:
            np.ndarray: bgr image representation.
        """
        # To accurately display colors, you need to choose constants

        im_r = hyper_image[:, :, CircleCv2Segmenter._wave_len(630, step, begin_wave_len)]
        im_g = hyper_image[:, :, CircleCv2Segmenter._wave_len(510, step, begin_wave_len)]
        im_b = hyper_image[:, :, CircleCv2Segmenter._wave_len(450, step, begin_wave_len)]

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

    @staticmethod
    def get_central_point(mask):
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
        max_contour = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]

        try:
            moments = cv2.moments(max_contour)

            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])

            central_point = (cx, cy)
        except:
            pass

        return central_point

    @staticmethod
    def get_max_contour_mask(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]
        return cv2.fillPoly(np.zeros(mask.shape, dtype='uint8'), pts=[max_contour], color=1)

    @staticmethod
    def get_radius(mask: np.ndarray) -> np.ndarray:
        try:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_contour = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]
            area = cv2.contourArea(max_contour)
        except:
            area = 0
        return np.sqrt(area / np.pi)

    def get_mask(self, hyper_image: np.ndarray,
                 step: int = 4, begin_wave_len: int = 450) -> np.ndarray:
        bgr = CircleCv2Segmenter._get_bgr(hyper_image, step, begin_wave_len)
        img_black = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, im_thr_circle = cv2.threshold(img_black, self.threshold_circle, 255, cv2.THRESH_BINARY)
        radius = CircleCv2Segmenter.get_radius(im_thr_circle)
        circle_mask = np.zeros_like(im_thr_circle)
        center = CircleCv2Segmenter.get_central_point(im_thr_circle)
        circle_mask = cv2.circle(circle_mask, center, int(2 * (radius / 3)), 1, -1)
        _, im_thr = cv2.threshold(img_black, self.threshold, 255, cv2.THRESH_BINARY)
        im_thr = im_thr.clip(0, 1)
        mask = im_thr * circle_mask
        return mask
