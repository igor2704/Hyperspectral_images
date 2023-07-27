from abc import ABC, abstractmethod

import numpy as np


class Segmenter(ABC):
    """
    Abstract class for hyperspectral image segmentation.
    """

    @abstractmethod
    def get_mask(self, hyper_image: np.ndarray) -> np.ndarray:
        """
        Args:
            hyper_image: multichannel hyperspectral image.
        Returns:
            np.ndarray: image mask.
        """
        pass
