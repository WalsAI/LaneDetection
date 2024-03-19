from torch.utils.data import Dataset
from typing import Optional, Callable
import pandas as pd
import cv2
import numpy as np

class RoadMarkingDataset(Dataset):
    """
        RoadMarkingDataset class for loading the dataset.
    """
    def __init__(self,
                 dataset_csv,
                 image_column,
                 mask_column,
                 transform: Optional[Callable] = None,
                 gt_transform: Optional[Callable] = None) -> None:
        """
        Initialization for RoadMarkingDataset.

        Args:
            dataset_csv: Path to the dataset csv file which contains path to image and corresponding mask.
            image_column: Name of the column in the csv file which contains the path to the image.
            mask_column: Name of the column in the csv file which contains the path to the mask.
            transform: Transformation to apply on the image (see torchvision.transforms).
            gt_transform: Transformation to apply on the mask (see torchvision.transforms).
        """
        self.dataset_csv = pd.read_csv(dataset_csv)
        self.image_column = image_column
        self.mask_column = mask_column
        self.transform = transform
        self.gt_transform = gt_transform
    
    def __len__(self):
        """
        Get the length of the dataset.
        """
        return self.dataset_csv.shape[0]

    def __getitem__(self, idx):
        """
        Get one item from the dataset.

        Args:
            idx: Index of the item to get from the dataset.
        """
        img_path = self.dataset_csv[self.image_column][idx]
        mask_path = self.dataset_csv[self.mask_column][idx]

        # RGB images
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # set -1 flag to read the image as it is
        mask = cv2.imread(mask_path, -1)

        if self.transform:
            image = self.transform(image)
        if self.gt_transform:
            mask = self.gt_transform(mask)
        
        return image, mask.astype(np.float32)

        
