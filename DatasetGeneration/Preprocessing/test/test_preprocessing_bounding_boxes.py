"""
    Tests for PreprocessingBoundingBoxes class.
"""

# local imports and paths addition
import sys
sys.path.append('..')
sys.path.append('Preprocessing')
sys.path.append('LaneDetection/DatasetGeneration/Preprocessing')
sys.path.append('/teamspace/studios/this_studio/LaneDetection/DatasetGeneration/Preprocessing')


# LaneDetection imports
from ProcessingPolygons import ProcessingPolygons
from LabelToImageConverter import LabelToImageConverter
import matplotlib.pyplot as plt

# third-party imports
import cv2
import numpy as np

def test_preprocessing_polygons():
    """
    Test for PreprocessingPolygons class.
    """

    dummy_label = '9a_jpg.rf.9e955d213a2b8656f81fb547616447f0.txt'
    dummy_label_folder = '/teamspace/studios/this_studio/LaneDetection/DatasetGeneration/Preprocessing/test/dummy_label_folder'
    dummy_images_folder = '/teamspace/studios/this_studio/LaneDetection/DatasetGeneration/Preprocessing/test/dummy_images_folder'
    dummy_masks_folder = '/teamspace/studios/this_studio/LaneDetection/DatasetGeneration/Preprocessing/test/dummy_masks_folder'
    preprocessing_polygons = ProcessingPolygons(dummy_label_folder, dummy_images_folder, dummy_masks_folder)

    # test _preprocess_file method
    polygons, classes = preprocessing_polygons._preprocess_file(dummy_label_folder + '/' + dummy_label, (640, 640))
    
    assert len(polygons) == 9

    assert len(classes) == 9

    preprocessing_polygons()

    mask = cv2.imread(dummy_masks_folder + '/' + LabelToImageConverter.LabelToImage(dummy_label, '.png'), -1)
    
    plt.imshow(mask)
    plt.savefig('mask_segmentation.png')

    assert mask.shape == (640, 640)
    assert np.unique(mask).tolist() == [0, 2, 5]