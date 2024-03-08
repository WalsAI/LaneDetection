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
    dummy_label_output = 'Preprocessing/test/dummy_label_output.csv'
    preprocessing_bounding_boxes = ProcessingPolygons(dummy_label, dummy_label_output)

    # test _preprocess_file method
    image_path, polygons, classes = preprocessing_bounding_boxes._preprocess_file(dummy_label)
    
    assert image_path == '9a_jpg.rf.9e955d213a2b8656f81fb547616447f0.jpg'
    assert len(polygons) == 9
    assert len(classes) == 9
    