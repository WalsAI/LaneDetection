import pytest

@pytest.fixture
def input_dict():
    return {
            'csv_path': 'dataset/valid/image_csv.csv',
            'images': 'images',
            'masks': 'masks'
           }


