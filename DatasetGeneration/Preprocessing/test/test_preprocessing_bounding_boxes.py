"""
    Tests for PreprocessingBoundingBoxes class.
"""

# local imports and paths addition
import sys
sys.path.append('Preprocessing/')

# LaneDetection imports
from ProcessingPolygons import ProcessingPolygons

# third-party imports
import pytest

def test_preprocessing_bounding_boxes():
    """
    Test for PreprocessingBoundingBoxes class.
    """

    dummy_label = '../dataset/test/labels/9a_jpg.rf.9e955d213a2b8656f81fb547616447f0.txt'
    dummy_folder = 'Preprocessing/test/dummy_label_folder'
    dummy_label_output = 'Preprocessing/test/dummy_label_output.csv'
    preprocessing_polygons = ProcessingPolygons(dummy_folder, dummy_label_output)

    # test _preprocess_file method
    image_path, polygons, classes = preprocessing_polygons._preprocess_file(dummy_label)
    
    assert image_path == 'Preprocessing/test/dummy_label_folder/9a_jpg.rf.9e955d213a2b8656f81fb547616447f0.jpg'
    assert len(polygons) == 9
    assert len(classes) == 9

    # test preprocess_gt method
    output_dict = preprocessing_polygons.preprocess_gt_folder()

    preprocessing_polygons()    