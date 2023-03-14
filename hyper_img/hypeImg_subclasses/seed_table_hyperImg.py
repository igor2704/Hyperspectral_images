from hyper_img.hyperImg import HyperImg

import numpy as np
import pandas as pd

import tifffile as tiff


class TableHyperImg(HyperImg):
    """
    Subclass of HyperImg for work with table annotations of hyperspectral images.
    """
    
    def __init__(self, path: str, table: pd.DataFrame,
                 threshold_value: float = 25,
                 savgol_par: tuple[int, int] = (9, 3),
                 target_varible_name: str = 'Target Varible',
                 name_column: str = 'Image Name',
                 black_calibr_name_column: str = 'Black calibration data',
                 white_calibr_name_column: str = 'White calibration data') -> None:
        """
        Args:
            path (str): images path.
            table (pd.DataFrame): table with columns: Name, self.target_varible_name,
                              Black image for calibration name, White image for calibration name.
            threshold_value (float, optional): threshold. Defaults to 25.
            savgol_par (tuple[int, int], optional): parametrs for savgol method. Defaults to (9, 3).
            target_varible_name (str, optional): name of target varible. Defaults to 'Target Varible'.
            black_calibration_img_path (str): black image for calibration path. Defaults to ''.
            white_calibration_img_path (str): white image for calibration path. Defaults to ''.
            name_column (str): column title with image name.
            black_calibr_name_column (str): column title with black image for calibration name.
            white_calibr_name_column (str): column title with white image for calibration name.
        """     
        self.table = table
        self.name_column = name_column
        self.black_calibr_name_column = black_calibr_name_column
        self.white_calibr_name_column = white_calibr_name_column
        super().__init__(path, threshold_value, savgol_par, target_varible_name)
        
    def _get_tiff(self) -> np.ndarray:
        if len(self.table[self.table[self.name_column] == self.path.split('/')[-1]]) == 0:
            raise NameError('Error in path')

        dir_name: str = '/'.join(self.path.split('/')[:-1]) + '/'
        img = tiff.imread(self.path)
        if self.table[self.table[self.name_column] == self.path.split('/')[-1]
                       ][self.black_calibr_name_column].iloc[0]:
            self.black_calibration_img_name = self.table[self.table[self.name_column] == self.path.split('/')[-1]
                                                         ][self.black_calibr_name_column].iloc[0]
            bl_img = tiff.imread(dir_name + self.black_calibration_img_name)
            new_img = np.where(bl_img > img, 0, img - bl_img)
        else:
            bl_img = np.zeros(img.shape)
            new_img = img    
        if not self.table[self.table[self.name_column] == self.path.split('/')[-1]
                          ][self.white_calibr_name_column].iloc[0]:
            return new_img
        self.white_calibration_img_name = self.table[self.table[self.name_column] == self.path.split('/')[-1]
                                                     ][self.white_calibr_name_column].iloc[0]
        wh_img = tiff.imread(dir_name + self.white_calibration_img_name)
        try:
            return new_img/(wh_img - bl_img)
        except ZeroDivisionError:
            raise ZeroDivisionError('black image is equal to white in one component')
    
    def _get_target_varible(self) -> str:
        return self.table[self.table[self.name_column] == self.path.split('/')[-1]
                          ][self.target_varible_name].iloc[0]
    
