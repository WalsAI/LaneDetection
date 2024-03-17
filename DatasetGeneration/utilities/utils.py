import logging

def configure_preprocessing_logger(mode: str = 'train'):
    logger = logging.getLogger('PreprocessingPolygons')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f'logs/generate_dataset_{mode}.log', mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.propagate = False        
    return logger