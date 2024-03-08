# local imports
from typing import Union, List, Dict

# third-party imports
import numpy as np
import os
import logging
import pandas as pd

# LaneDetection imports
from LabelToImageConverter import LabelToImageConverter

logging.getLogger('ProcessingPolygons')

class ProcessingPolygons:
    """
    Preprocessing polygons from dataset in order to generate masks with Segment Anything Model (SAM) as states in `https://arxiv.org/pdf/2304.02643.pdf`.
    """
    def __init__(
        self,
        path_to_dataset: str,
        path_to_output_csv: str,
    ) -> None:
        """
        Initialization for preprocessing the polygons in order
        to match Segment Anything Model (SAM) as states in LabelAnything repository `https://github.com/WalsAI/LabelAnything`.
        
        Args:
            path_to_dataset: Path to directory for labels (train/val/test).
            path_to_output_csv: Path to output csv file in format suggested in LabelAnything `https://github.com/WalsAI/LabelAnything`
        """
        self.path_to_dataset = path_to_dataset
        self.path_to_output_csv = path_to_output_csv

    def _preprocess_file(self, file_path: str) -> Union[str, List[float], List[float]]:
        """
            Preprocess ground truth file and then add it to csv file.

            Args:
                file_path: Path to file for preprocessing.
        """
        image_path = LabelToImageConverter.LabelToImage(file_path)
        
        # list of polygon coordinates
        polygons = list()

        # list of classes
        classes = list()

        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                line_split = line.split(' ')

                # store in a variabile in order to have the same format
                cls = int(line_split[0])
                polygon = line_split[1:]
                polygon = [float(p) for p in polygon]

                polygons.append(polygon)
                classes.append(cls)
                
        return image_path, polygons, classes
    
    # TODO: have to check if this is the expected format
    def preprocess_gt_folder(self) -> Dict[str, Union[List[str], List[np.ndarray], List[np.ndarray]]]:
        """
        Preprocess folder containing polygons and classes as labels.
        """
        labels = sorted(os.listdir(self.path_to_dataset))
        
        image_paths = list()
        polygons = list()
        classes = list()
        for label in labels:
            logging.info(f'Currently processing {label}...')
            img_path, polygon, cls = self._preprocess_file(self.path_to_dataset + '/' + label)
            image_paths.append(img_path)
            polygons.append(polygon)
            classes.append(cls)
        
        return {
            'Image_Paths': image_paths,
            'polygons': polygons,
            'classes': classes
        }

    def df_to_csv(self, df: Dict[str, Union[List[str], List[np.ndarray], List[np.ndarray]]]) -> None:
        """
        Convert dictionary to csv file.

        Args:
            df: Dictionary containing the keys 'Image_Paths', 'polygons', 'classes'.
        """
        df = pd.DataFrame(df)
        df.to_csv(self.path_to_output_csv, index=False)
        logging.info(f'CSV file saved at {self.path_to_output_csv}')
    
    def __call__(self) -> None:
        """
        Preprocess polygons and classes and then save them to csv file.
        """
        df = self.preprocess_gt_folder()
        self.df_to_csv(df)
            
                
            



