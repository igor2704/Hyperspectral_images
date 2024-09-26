from hyper_img.hyperImg import HyperImg
from hyper_img.hyperImg_segmenter.segmenter import Segmenter

import numpy as np
import pandas as pd
import typing as tp

import tifffile as tiff


class NormalizationError(Exception):
    pass


class TableHyperImg(HyperImg):
    """
    Subclass of HyperImg for work with table annotations of hyperspectral images.
    """

    def __init__(self,
                 path: str, 
                 table: pd.DataFrame,
                 segmenter: Segmenter,
                 savgol_par: tuple[int, int] = (9, 3),
                 name_column: str = 'Image Name',
                 black_calibr_name_column: str = 'Black calibration data',
                 white_calibr_name_column: str = 'White calibration data',
                 plant_number_column: str = 'PlantNumber',
                 id_name_column: str = 'ID партии',
                 factors_columns: tp.Sequence[str] | None = None,
                 column_for_marking: str = '') -> None:
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
            column_for_marking (str): if not '', mark the row in the table after reading. Defaults 'Marking'.
        """
        if len(table[table[name_column] == path.split('/')[-1]]) == 0:
            raise NameError('Error in path')
        if column_for_marking != '' and column_for_marking not in table.columns:
            table[column_for_marking] = len(table[name_column]) * [0]
        if column_for_marking == '':
            self.row = table[table[name_column] == path.split('/')[-1]].iloc[0]
        else:
            try:
                self.row = table[(table[name_column] == path.split('/')[-1]) &
                                    (table[column_for_marking] == 0)].iloc[0]
            except IndexError:
                raise IndexError(f'try delete column: "{column_for_marking}" in table')
            index = np.arange(len(table))[(table[name_column] == path.split('/')[-1]) &
                                  (table[column_for_marking] == 0)][0]
            table.loc[index, column_for_marking] = 1
        if column_for_marking != '' and len(table) == len(table[table[column_for_marking] == 1]):
            table = table.drop(columns=[column_for_marking])
        self.name_column = name_column
        self.plant_column = plant_number_column
        self.black_calibr_name_column = black_calibr_name_column
        self.white_calibr_name_column = white_calibr_name_column
        self.id_column = id_name_column
        factors = dict()
        for col in factors_columns:
            factors[col] = self.row[col]
        super().__init__(path, segmenter, savgol_par, factors=factors)

    def _get_image(self) -> np.ndarray:
        dir_name: str = '/'.join(self.path.split('/')[:-1]) + '/'
        self.dir_name = dir_name
        img = tiff.imread(self.path).astype(np.float64)
        self.object_name = self.row[self.plant_column]
        if self.row[self.black_calibr_name_column]:
            self.black_calibration_img_name = self.row[self.black_calibr_name_column]
            self.bl_img = tiff.imread(dir_name + self.black_calibration_img_name).astype(np.float64)
            new_img = img - self.bl_img
        else:
            self.bl_img = np.zeros(img.shape, dtype=np.float64)
            new_img = img
        if self.row[self.white_calibr_name_column]:
            self.white_calibration_img_name = self.row[self.white_calibr_name_column]
            self.wh_img = tiff.imread(dir_name + self.white_calibration_img_name).astype(np.float64)
            try:
                new_img = new_img/(self.wh_img - self.bl_img)
            except ZeroDivisionError:
                raise ZeroDivisionError('black image is equal to white in one component')
        else:
            self.wh_img = np.zeros(img.shape, dtype=np.float64)
        return new_img
