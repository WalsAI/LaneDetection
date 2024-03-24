import logging
import torch
import numpy as np

def configure_building_csv_logger(mode: str = 'train'):
    logger = logging.getLogger('BuildCSVFromDir')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f'logs/building_csv_{mode}.log', mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.propagate = False        
    return logger

def nd_array_to_torch_float_channels_first(image: np.ndarray) -> torch.Tensor:
    """
    Convert a numpy array to a torch tensor with channels first.
    """
    return torch.from_numpy(image).float().permute(2, 0, 1)