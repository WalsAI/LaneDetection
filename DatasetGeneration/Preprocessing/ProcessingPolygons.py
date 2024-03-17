# local imports
from typing import Union, List, Dict, Tuple

# third-party imports
import numpy as np
import os
import logging
import pandas as pd
import cv2

# LaneDetection imports
from DatasetGeneration.utilities.utils import configure_preprocessing_logger
from LabelToImageConverter import LabelToImageConverter

class ProcessingPolygons:
    """
    Preprocessing polygons from dataset to convert them to masks.
    """
    def __init__(
        self,
        path_to_labels: str,
        path_to_images: str,
        path_to_masks: str,
        mode: str = 'train'
    ) -> None:
        """
        Initialization for preprocessing the polygons to prepare them for conversion to actual masks.
        
        Args:
            path_to_labels: Path to directory for labels.
            path_to_images: Path to directory for images.
            path_to_masks: Path to directory for masks.
            mode: Mode of the dataset (i.e. train, val, test).
            
        """
        self.path_to_labels = path_to_labels
        self.path_to_images = path_to_images
        self.path_to_masks = path_to_masks
        self.logger = configure_preprocessing_logger(mode)
        self.mode = mode

    @staticmethod
    def _get_image_size(image_path: str) -> Tuple[int]:
        """
        Get the size of the image.

        Args:
            image_path: Path to image for getting the size.
        """
        image = cv2.imread(image_path)
        return image.shape[:2]

    @staticmethod
    def _postprocess_polygon(polygon: List[str], image_size: Tuple[int]) -> np.ndarray:
        """
        Postprocess the polygon to fit the image size.

        Args:
            polygon: List of polygon coordinates.
            image_size: Size of the image.
        """
        processed_polygon = []
        for i in range(0, len(polygon), 2):
            point = polygon[i:i+2]
            # cv2 uses (height, width) format
            point = [float(point[0]) * image_size[1], float(point[1]) * image_size[0]]
            processed_polygon.append(point)
        return np.array(processed_polygon)

    def _preprocess_file(self, file_path: str, image_size: Tuple[int]) -> Union[str, List[List[float]], List[float]]:
        """
            Preprocess ground truth file and then add it to csv file.

            Args:
                file_path: Path to file for preprocessing.
                image_size: Size of the image.
        """        
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
                polygon = self._postprocess_polygon(polygon, image_size)

                polygons.append(polygon)
                classes.append(cls)
                
        return polygons, classes
    
    @staticmethod
    def _get_mask_from_polygons(polygons: List[np.ndarray], labels: List[int], image_size) -> np.ndarray:
        """
        Get mask from polygons and labels.

        Args:
            polygons: List of polygons.
            labels: List of labels.
            image_size: Size of the image.
        """
        masks = []
        for i in range(len(polygons)):
            mask = np.zeros(image_size)
            mask = cv2.fillPoly(mask, np.int32([polygons[i]]), int(labels[i]))
            masks.append(mask)
        return masks

    @staticmethod
    def _concatenate_masks(masks: List[np.ndarray], image_size: Tuple[int]):
        """
        Concatenate masks.

        Args:
            masks: List of masks.
            image_size: Size of the image.
        """
        # All the masks should have the same shape since they are from the same image
        output = np.zeros(image_size)
        for mask in masks:
            output = np.where(output == 0, mask, output)
        return output

    def preprocess_gt_folder(self) -> Dict[str, Union[List[str], List[List[List[str]]], List[np.ndarray]]]:
        """
        Preprocess folder containing polygons and classes as labels and transform the polygons into segmentation GT.
        """
        labels = sorted(os.listdir(self.path_to_labels))
        self.logger.info(f'Generating dataset for {self.mode}')
        for label in labels:
            self.logger.info(f'Currently processing {label}...')
            image_path = self.path_to_images + '/' + LabelToImageConverter.LabelToImage(label, '.jpg')
            image_size = self._get_image_size(image_path)
            polygon, cls = self._preprocess_file(self.path_to_labels + '/' + label, image_size)
            # convert polygons and classes into a segmentation mask stored in path_to_masks folder
            mask = self._get_mask_from_polygons(polygon, cls, image_size)
            mask = self._concatenate_masks(mask, image_size)
            self._write_mask(mask, self.path_to_masks, self._get_image_title(image_path, '.png'))
        self.logger.info(f'Finished generating dataset for {self.mode}')

    @staticmethod
    def _get_image_title(image_path: str, format: str) -> str:
        """
        Get image title from image path.

        Args:
            image_path: Path to image.
        """
        return image_path.split('/')[-1][:-4] + format

    @staticmethod
    def _write_mask(mask: np.ndarray, path_to_mask: str, image_title: str) -> None:
        """
        Write mask to file.

        Args:
            mask: GT mask.
            path_to_mask: Path to mask.
            image_title: title of the current image.
        """
        cv2.imwrite(path_to_mask + '/' + image_title, mask)
        
    
    def __call__(self) -> None:
        """
        Preprocess polygons and classes and then save them to csv file.
        """
        self.preprocess_gt_folder()
            
                
            



