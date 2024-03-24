from LaneDetection.Data.Dataset import RoadMarkingDataset
from LaneDetection.Data.DataLoader import RoadMarkingDataloader


def test_dataloader(input_dict):
    """
    Test for Dataloader and Dataset class.
    """

    dataloader = RoadMarkingDataloader(input_dict['csv_path'], input_dict['csv_path'], None, 1, input_dict['images'], input_dict['masks'], [104.91310576425163, 109.82240221886539, 108.90190913395513], [29.300869498373267, 30.481186465060542, 31.239792631061004])
    dataloader.setup('fit')

    train_loader = dataloader.train_dataloader()
    image, mask = next(iter(train_loader))
    print(image.shape)
    assert image.shape == (1, 3, 640, 640)
    assert mask.shape == (1, 1, 640, 640)



