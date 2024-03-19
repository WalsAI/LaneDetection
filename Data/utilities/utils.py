import logging

def configure_building_csv_logger(mode: str = 'train'):
    logger = logging.getLogger('BuildCSVFromDir')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f'../../../logs/building_csv_{mode}.log', mode='w')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.propagate = False        
    return logger