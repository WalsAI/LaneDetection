from utilities.utils import configure_building_csv_logger
import os
import pandas as pd
import cv2
import numpy as np

class BuildCSVFromDir:    
    """
    This class is used to build a CSV file from a directory of images and masks.
    """
    def __init__(self, path_to_images: str, path_to_masks: str, path_to_csv: str, mode: str = 'train'):
        """
        Initialization for BuildCSVFromDir.

        Args:
            path_to_images: Path to directory for images.
            path_to_masks: Path to directory for masks.
            path_to_csv: Path to the csv file to be created.
        """
        self.path_to_images = path_to_images
        self.path_to_masks = path_to_masks
        self.path_to_csv = path_to_csv
        self.logger = configure_building_csv_logger(mode)
    
    def build_csv(self):
        """
        Build the csv file from the directory of images and masks.
        """
        images = os.listdir(self.path_to_images)
        masks = os.listdir(self.path_to_masks)
        images.sort()
        masks.sort()
        data = {'image': [], 'mask': []}
        for image, mask in zip(images, masks):
            data['image'].append(os.path.join(self.path_to_images, image))
            data['mask'].append(os.path.join(self.path_to_masks, mask))
        df = pd.DataFrame(data)
        df.to_csv(self.path_to_csv, index=False)
        self.logger.info(f"CSV file created at {self.path_to_csv}")
    
    def __call__(self):
        """
        Call the build_csv method.
        """
        self.build_csv()