from hyper_img.hyperImg import HyperImg
from hyper_img.hyperImg_segmenter.segmenter import Segmenter

import numpy as np
import pandas as pd

import tifffile as tiff


class NormalizationError(Exception):
    pass


class TableHyperImg(HyperImg):
    """
    Subclass of HyperImg for work with table annotations of hyperspectral images.
    """
    
    def __init__(self, path: str, table: pd.DataFrame,
                 segmenter: Segmenter,
                 savgol_par: tuple[int, int] = (9, 3),
                 target_varible_name: str = 'Target Varible',
                 name_column: str = 'Image Name',
                 black_calibr_name_column: str = 'Black calibration data',
                 white_calibr_name_column: str = 'White calibration data',
                 plant_number_column: str = 'PlantNumber',
                 id_name_column: str = 'ID партии',
                 with_median: bool = True) -> None:
        """
        Args:
            path (str): images path.
            table (pd.DataFrame): table with columns: Name, self.target_varible_name,
                                 Black image for calibration name, White image for calibration name.
            segmenter (Segmenter): object for segmentation.
            savgol_par (tuple[int, int], optional): parametrs for savgol method. Defaults to (9, 3).
            target_varible_name (str, optional): name of target varible. Defaults to 'Target Varible'.
            black_calibration_img_path (str): black image for calibration path. Defaults to ''.
            white_calibration_img_path (str): white image for calibration path. Defaults to ''.
            name_column (str): column title with image name.
            black_calibr_name_column (str): column title with black image for calibration name.
            white_calibr_name_column (str): column title with white image for calibration name.
            with_median (bool): if true, then calculate medians for each channel. Defaults True.
        """     
        self.table = table
        self.name_column = name_column
        self.plant_column = plant_number_column
        self.black_calibr_name_column = black_calibr_name_column
        self.white_calibr_name_column = white_calibr_name_column
        self.id_column = id_name_column
        super().__init__(path, segmenter, savgol_par, target_varible_name, with_median=with_median)
        
    def _get_image(self) -> np.ndarray:
        if len(self.table[self.table[self.name_column] == self.path.split('/')[-1]]) == 0:
            raise NameError('Error in path')
        dir_name: str = '/'.join(self.path.split('/')[:-1]) + '/'
        self.dir_name = dir_name
        img = tiff.imread(self.path).astype(np.float64)
        self.object_name = self.table[self.table[self.name_column] == self.path.split('/')[-1]
                                      ][self.plant_column].iloc[0]
        if self.table[self.table[self.name_column] == self.path.split('/')[-1]
                      ][self.black_calibr_name_column].iloc[0]:
            self.black_calibration_img_name = self.table[self.table[self.name_column] == self.path.split('/')[-1]
                                                         ][self.black_calibr_name_column].iloc[0]
            self.bl_img = tiff.imread(dir_name + self.black_calibration_img_name).astype(np.float64)
            new_img = img - self.bl_img
        else:
            self.bl_img = np.zeros(img.shape, dtype=np.float64)
            new_img = img    
        if self.table[self.table[self.name_column] == self.path.split('/')[-1]
                      ][self.white_calibr_name_column].iloc[0]:
            self.white_calibration_img_name = self.table[self.table[self.name_column] == self.path.split('/')[-1]
                                                         ][self.white_calibr_name_column].iloc[0]
            self.wh_img = tiff.imread(dir_name + self.white_calibration_img_name).astype(np.float64)
            try:
                new_img = new_img/(self.wh_img - self.bl_img)
            except ZeroDivisionError:
                raise ZeroDivisionError('black image is equal to white in one component')
        else:
            self.wh_img = np.zeros(img.shape, dtype=np.float64)
        return new_img

    def _get_target_varible(self) -> str:
        return self.table[self.table[self.name_column] == self.path.split('/')[-1]
                          ][self.target_varible_name].iloc[0]
    
